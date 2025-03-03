use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{DeviceOwned, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline},
    sync::GpuFuture,
    DeviceSize,
};

use super::MyVertex;

const START_CAPACITY: u32 = 1024;

pub struct DynMemoryManager {
    pub(crate) device_local_buffer: Subbuffer<[MyVertex]>,
    pub(crate) descriptor_set: Arc<DescriptorSet>,
    size: u32,
    capacity: u32,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
    compute_pipeline: Arc<ComputePipeline>,
}

impl DynMemoryManager {
    // starts with 0 pixels, and a capacity of 1024.
    pub fn new(
        memory_allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        queue: Arc<Queue>,
        compute_pipeline: Arc<ComputePipeline>,
    ) -> Self {
        let device_local_buffer = Buffer::new_slice::<MyVertex>(
            memory_allocator.clone(),
            BufferCreateInfo {
                // Specify use as a storage buffer, vertex buffer, and transfer destination.
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // Specify this buffer will only be used by the device.
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            START_CAPACITY as DeviceSize,
        )
        .unwrap();

        // initialize descriptor_set
        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            // 0 is the index of the descriptor set.
            compute_pipeline.layout().set_layouts()[0].clone(),
            [
                // 0 is the binding of the data in this set. We bind the `Buffer` of vertices here.
                WriteDescriptorSet::buffer(0, device_local_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        Self {
            device_local_buffer,
            descriptor_set,
            size: 0,
            capacity: START_CAPACITY,
            memory_allocator: memory_allocator.clone(),
            descriptor_set_allocator: descriptor_set_allocator.clone(),
            command_buffer_allocator: command_buffer_allocator.clone(),
            queue: queue.clone(),
            compute_pipeline: compute_pipeline.clone(),
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn add_pixels(&mut self, num_pixels: u32) {
        if self.size + num_pixels > self.capacity {
            self.recreate_buffer(std::cmp::max(self.capacity * 2, self.size + num_pixels));
        }

        let staging_buffer = Buffer::new_slice::<MyVertex>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            num_pixels as DeviceSize,
        )
        .unwrap();

        {
            let mut mapping = staging_buffer.write().unwrap();
            for vertex in mapping.iter_mut() {
                *vertex = MyVertex {
                    pos: [0.0, 0.0],
                    vel: [0.0, 0.0],
                };
            }
        }

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer.clone(),
                self.device_local_buffer
                    .clone()
                    .slice(self.size as DeviceSize..(self.size + num_pixels) as DeviceSize),
            ))
            .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        let device = self.memory_allocator.device();

        let future = vulkano::sync::now(device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        self.size += num_pixels;
    }

    fn recreate_buffer(&mut self, new_capacity: u32) {
        println!("changing??? new capacity is: {}", new_capacity);
        let new_buffer = Buffer::new_slice::<MyVertex>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            new_capacity as DeviceSize,
        )
        .unwrap();

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.device_local_buffer.clone(),
                new_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        let device = self.memory_allocator.device();

        let future = vulkano::sync::now(device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        // now recreate descriptor_set
        let new_descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            // 0 is the index of the descriptor set.
            self.compute_pipeline.layout().set_layouts()[0].clone(),
            [
                // 0 is the binding of the data in this set. We bind the `Buffer` of vertices here.
                WriteDescriptorSet::buffer(0, new_buffer.clone()),
            ],
            [],
        )
        .unwrap();
        println!(
            "descriptor_set changed to a mindboggling size of: {}",
            self.device_local_buffer.len()
        );

        self.descriptor_set = new_descriptor_set;

        self.device_local_buffer = new_buffer;
        self.capacity = new_capacity;
    }

    #[allow(dead_code)]
    pub fn debug_buffer(&self) {
        let debug_buffer = Buffer::new_slice::<MyVertex>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST,
                ..Default::default()
            },
            self.size as DeviceSize,
        )
        .unwrap();

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer_builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.device_local_buffer.clone(),
                debug_buffer.clone(),
            ))
            .unwrap();

        let command_buffer = command_buffer_builder.build().unwrap();

        let device = self.memory_allocator.device();

        let future = vulkano::sync::now(device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let data = debug_buffer.read().unwrap();
        for (i, vertex) in data.iter().enumerate() {
            println!("Vertex {}: {:?}", i, vertex);
        }
    }
}
