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

            // Allow push constants to define a parameters of compute.
            layout (push_constant) uniform PushConstants {
                vec2 attractor;
                float attractor_strength;
                float delta_time;
            } push;

            // Keep this value in sync with the `maxSpeed` const in the vertex shader.
            const float maxSpeed = 10.0;

            const float minLength = 0.02;
            const float friction = -2.0;

            void main() {
                const uint index = gl_GlobalInvocationID.x;

                vec2 vel = vertices[index].vel;

                // Update particle position according to velocity.
                vec2 pos = vertices[index].pos + push.delta_time * vel;

                // Bounce particle off screen-border.
                if (abs(pos.x) > 1.0) {
                    vel.x = sign(pos.x) * (-0.95 * abs(vel.x) - 0.0001);
                    if (abs(pos.x) >= 1.05) {
                        pos.x = sign(pos.x);
                    }
                }
                if (abs(pos.y) > 1.0) {
                    vel.y = sign(pos.y) * (-0.95 * abs(vel.y) - 0.0001);
                    if (abs(pos.y) >= 1.05) {
                        pos.y = sign(pos.y);
                    }
                }

                // Simple inverse-square force.
                vec2 t = push.attractor - pos;
                float r = max(length(t), minLength);
                vec2 force = push.attractor_strength * (t / r) / (r * r);

                // Update velocity, enforcing a maximum speed.
                vel += push.delta_time * force;
                if (length(vel) > maxSpeed) {
                    vel = maxSpeed*normalize(vel);
                }

                // Set new values back into buffer.
                vertices[index].pos = pos;
                vertices[index].vel = vel * exp(friction * push.delta_time);
            }
        ",
    }
}
