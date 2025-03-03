// The vertex shader determines color and is run once per particle. The vertices will be
// updated by the compute shader each frame.
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) in vec2 pos;
            layout(location = 1) in vec2 vel;

            layout(location = 0) out vec4 outColor;

            // Keep this value in sync with the `maxSpeed` const in the compute shader.
            const float maxSpeed = 10.0;

            void main() {
                gl_Position = vec4(pos, 0.0, 1.0);
                gl_PointSize = 1.0;

                // Mix colors based on position and velocity.
                outColor = mix(
                    0.2 * vec4(pos, abs(vel.x) + abs(vel.y), 1.0),
                    vec4(1.0, 0.5, 0.8, 1.0),
                    sqrt(length(vel) / maxSpeed)
                );
            }
        ",
    }
}

// The fragment shader will only need to apply the color forwarded by the vertex shader,
// because the color of a particle should be identical over all pixels.
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec4 outColor;

            layout(location = 0) out vec4 fragColor;

            void main() {
                fragColor = outColor;
            }
        ",
    }
}

// Compute shader for updating the position and velocity of each particle every frame.
pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450

            layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

            struct VertexData {
                vec2 pos;
                vec2 vel;
            };

            // Storage buffer binding, which we optimize by using a DeviceLocalBuffer.
            layout (binding = 0) buffer VertexBuffer {
                VertexData vertices[];
            };

            // Allow push constants to define parameters of compute.
            layout (push_constant) uniform PushConstants {
                float delta_time;
            } push;

            const float maxSpeed = 10.0;
            const float friction = -2.0;
            const float gravity = 9.8; // Constant gravity force downwards
            const float ground_damping = 0.7; // Controls how much speed is lost on hitting the ground

            void main() {
                const uint index = gl_GlobalInvocationID.x;

                vec2 vel = vertices[index].vel;

                // Update velocity with gravity (pulling downwards)
                vel.y += push.delta_time * gravity;

                // Update position
                vec2 pos = vertices[index].pos + push.delta_time * vel;

                // Bounce particle off screen borders.
                if (abs(pos.x) > 1.0) {
                    vel.x = sign(pos.x) * (-0.95 * abs(vel.x) - 0.0001);
                    pos.x = clamp(pos.x, -1.0, 1.0);
                }

                if (pos.y < -1.0) {
                    // Hitting the ground: dampen vertical velocity
                    vel.y *= -ground_damping;
                    pos.y = -1.0;
                } else if (pos.y > 1.0) {
                    // Top border: prevent escape
                    vel.y = -0.95 * abs(vel.y);
                    pos.y = 1.0;
                }

                // Apply friction
                vel *= exp(friction * push.delta_time);

                // Enforce max speed
                if (length(vel) > maxSpeed) {
                    vel = maxSpeed * normalize(vel);
                }

                // Store updated values
                vertices[index].pos = pos;
                vertices[index].vel = vel;
            }

        ",
    }
}
