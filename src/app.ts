import tgpu from "typegpu";
import { struct, vec2f, vec4f, arrayOf } from "typegpu/data";
import { wgsl } from "./wgsl";

const Vertex = struct({
  position: vec2f,
  color: vec4f,
});

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
  });

  const vertexBuffer = root.createBuffer(arrayOf(Vertex, 3), [
    {
      position: vec2f(0, 1),
      color: vec4f(1, 0, 0, 1),
    },
    {
      position: vec2f(1, -1),
      color: vec4f(0, 1, 0, 1),
    },
    {
      position: vec2f(-1, -1),
      color: vec4f(0, 0, 1, 1),
    },
  ]).$usage('storage');

  const renderModule = root.device.createShaderModule({
    code: wgsl/*wgsl*/ `
      @group(0) @binding(0) var<storage, read> vertices: array<Vertex>;

      struct Vertex {
        position: vec2<f32>,
        color: vec4<f32>,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
      };
      
      @vertex fn vs(
        @builtin(vertex_index) vertexIndex : u32,
      ) -> VertexOutput {
        let vertex = vertices[vertexIndex];
        return VertexOutput(
          vec4<f32>(vertex.position, 0.0, 1.0),
          vertex.color
        );
      }

      @fragment fn fs(
        input: VertexOutput,
      ) -> @location(0) vec4<f32> {
        return input.color;
      }
    `,
  });

  const renderBindGroupLayout = tgpu.bindGroupLayout({
    buffer: { storage: vertexBuffer.dataType },
  });

  const renderBindGroup = renderBindGroupLayout.populate({
    buffer: vertexBuffer,
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
      targets: [{ format }],
    },
  });

  const renderPassDescriptor = {
    label: "Render pass descriptor",
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: [0.3, 0.3, 0.3, 1.0],
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  } satisfies GPURenderPassDescriptor;

  const handleResize = () => {
    canvas.width = Math.ceil(window.innerWidth * window.devicePixelRatio);
    canvas.height = Math.ceil(window.innerHeight * window.devicePixelRatio);
  };
  window.addEventListener("resize", handleResize);
  handleResize();

  const handleFrame = () => {
    renderPassDescriptor.colorAttachments[0].view = context
      .getCurrentTexture()
      .createView();

    const encoder = root.device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass(renderPassDescriptor);
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, root.unwrap(renderBindGroup));
    renderPass.draw(3);
    renderPass.end();
    
    root.device.queue.submit([encoder.finish()]);

    requestAnimationFrame(handleFrame);
  };
  requestAnimationFrame(handleFrame);
}
