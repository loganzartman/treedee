import tgpu from "typegpu";
import { struct, vec3f, vec4f, mat4x4f, f32, arrayOf } from "typegpu/data";
import { mat4, utils } from "wgpu-matrix";
import { wgsl } from "./wgsl";

const Blob = struct({
  position: vec3f,
  radius: f32,
  color: vec4f,
}).$name("Blob");

const Uniforms = struct({
  cameraPos: vec3f,
  lookPos: vec3f,
  upVec: vec3f,
  cameraMat: mat4x4f,
  projMat: mat4x4f,
}).$name("Uniforms");

export async function init({ container }: { container: HTMLDivElement }) {
  const root = await tgpu.init();
  const canvas = container.querySelector<HTMLCanvasElement>("#app-canvas")!;
  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new Error("Failed to get webgpu context");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: root.device,
    format,
    alphaMode: "premultiplied",
  });

  const uniformsBuffer = root.createBuffer(Uniforms).$usage("uniform");

  const nBlobs = 1000000;
  const blobBuffer = root.createBuffer(arrayOf(Blob, nBlobs)).$usage("storage");
  const randomizeBlobs = () => {
    const blobs = Array.from({ length: nBlobs }, () => ({
      position: vec3f(
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
        Math.random() * 2 - 1,
      ),
      radius: Math.random() * 0.1,
      color: vec4f(0, 1, 0, 0.1),
    }));
    blobBuffer.write(blobs);
  };
  randomizeBlobs();

  const renderModule = root.device.createShaderModule({
    code: wgsl`
      @group(0) @binding(0) var<uniform> uniforms: Uniforms;
      @group(0) @binding(1) var<storage, read> blobs: array<Blob>;

      struct Uniforms {
        cameraPos: vec3<f32>,
        lookPos: vec3<f32>,
        upVec: vec3<f32>,
        cameraMat: mat4x4<f32>,
        projMat: mat4x4<f32>,
      };

      struct Blob {
        position: vec3<f32>,
        radius: f32,
        color: vec4<f32>,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
      };

      @vertex fn vs(
        @builtin(vertex_index) vertexIndex : u32,
      ) -> VertexOutput {
        let blob = blobs[vertexIndex];
        let pos = uniforms.projMat * uniforms.cameraMat * vec4f(blob.position, 1.0);
        return VertexOutput(pos, blob.color);
      }

      @fragment fn fs(
        input: VertexOutput,
      ) -> @location(0) vec4<f32> {
        return vec4f(input.color.rgb * input.color.a, input.color.a);
      }
    `,
  });

  const renderBindGroupLayout = tgpu.bindGroupLayout({
    uniforms: { uniform: uniformsBuffer.dataType },
    blobs: { storage: blobBuffer.dataType },
  });

  const renderBindGroup = renderBindGroupLayout.populate({
    uniforms: uniformsBuffer,
    blobs: blobBuffer,
  });

  const renderPipeline = root.device.createRenderPipeline({
    layout: root.device.createPipelineLayout({
      bindGroupLayouts: [root.unwrap(renderBindGroupLayout)],
    }),
    vertex: {
      module: renderModule,
    },
    fragment: {
      module: renderModule,
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "point-list",
    },
  });

  const renderPassDescriptor = {
    label: "Render pass descriptor",
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0.0, 0.0, 0.0, 1.0],
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  } satisfies GPURenderPassDescriptor;

  let aspectRatio = 1;
  const handleResize = () => {
    canvas.width = Math.ceil(window.innerWidth * window.devicePixelRatio);
    canvas.height = Math.ceil(window.innerHeight * window.devicePixelRatio);
    aspectRatio = canvas.width / canvas.height;
  };
  window.addEventListener("resize", handleResize);
  handleResize();

  const makeUniforms = () => {
    const cameraPos = vec3f(-5, 0, 3);
    const lookPos = vec3f(0, 0, 0);
    const upVec = vec3f(0, 0, 1);
    const cameraMat = mat4.lookAt(cameraPos, lookPos, upVec, mat4x4f());
    const projMat = mat4.perspective(
      utils.degToRad(100),
      aspectRatio,
      0,
      1000,
      mat4x4f(),
    );
    return { cameraPos, lookPos, upVec, cameraMat, projMat };
  };

  const handleFrame = () => {
    uniformsBuffer.write(makeUniforms());

    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const encoder = root.device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, root.unwrap(renderBindGroup));
    renderPass.draw(nBlobs);
    renderPass.end();

    root.device.queue.submit([encoder.finish()]);

    requestAnimationFrame(handleFrame);
  };
  requestAnimationFrame(handleFrame);
}
