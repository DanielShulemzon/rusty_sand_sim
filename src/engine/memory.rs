use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo,
    },
    device::{DeviceOwned, Queue},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
        StandardMemoryAllocator,
    },
    sync::GpuFuture,
    DeviceSize,
};

use super::MyVertex;

const START_CAPACITY: u32 = 1024;

pub struct DynVertexBuffer {
    pub(crate) device_local_buffer: Subbuffer<[MyVertex]>,
    size: u32,
    capacity: u32,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    queue: Arc<Queue>,
}

impl DynVertexBuffer {
    // starts with 0 pixels, and a capacity of 1024.
    pub fn new(
        memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
        queue: Arc<Queue>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
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

        Self {
            device_local_buffer,
            size: 0,
            capacity: START_CAPACITY,
            memory_allocator: memory_allocator.clone(),
            command_buffer_allocator: command_buffer_allocator.clone(),
            queue: queue.clone(),
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn get_buffer_clone(&self) -> Subbuffer<[MyVertex]> {
        self.device_local_buffer.clone()
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
