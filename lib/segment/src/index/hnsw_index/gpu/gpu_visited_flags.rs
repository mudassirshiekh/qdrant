use std::collections::HashMap;
use std::sync::Arc;

use common::types::PointOffsetType;

use super::shader_builder::ShaderBuilderParameters;
use super::GPU_TIMEOUT;
use crate::common::operation_error::OperationResult;

#[repr(C)]
pub struct GpuVisitedFlagsParamsBuffer {
    pub generation: u32,
}

pub struct GpuVisitedFlags {
    params: GpuVisitedFlagsParamsBuffer,
    params_buffer: Arc<gpu::Buffer>,
    params_staging_buffer: Arc<gpu::Buffer>,
    visited_flags_buffer: Arc<gpu::Buffer>,
    descriptor_set_layout: Arc<gpu::DescriptorSetLayout>,
    descriptor_set: Arc<gpu::DescriptorSet>,
    capacity: usize,
    remap: bool,
}

impl ShaderBuilderParameters for GpuVisitedFlags {
    fn shader_includes(&self) -> HashMap<String, String> {
        HashMap::from([(
            "visited_flags.comp".to_string(),
            include_str!("shaders/visited_flags.comp").to_string(),
        )])
    }

    fn shader_defines(&self) -> HashMap<String, Option<String>> {
        let mut defines = HashMap::new();
        defines.insert(
            "VISITED_FLAGS_CAPACITY".to_owned(),
            Some(self.capacity.to_string()),
        );
        if self.remap {
            defines.insert("VISITED_FLAGS_REMAP".to_owned(), None);
        }
        defines
    }
}

impl GpuVisitedFlags {
    pub fn new(
        device: Arc<gpu::Device>,
        groups_count: usize,
        points_remap: &[PointOffsetType],
    ) -> OperationResult<Self> {
        let alignment = std::mem::size_of::<u32>();
        let points_count = points_remap.len().next_multiple_of(alignment);

        let params_buffer = gpu::Buffer::new(
            device.clone(),
            "Visited flags params buffer",
            gpu::BufferType::Uniform,
            std::mem::size_of::<GpuVisitedFlagsParamsBuffer>(),
        )?;
        let params_staging_buffer = gpu::Buffer::new(
            device.clone(),
            "Visited flags params staging buffer",
            gpu::BufferType::CpuToGpu,
            std::mem::size_of::<GpuVisitedFlagsParamsBuffer>(),
        )?;

        let (visited_flags_buffer, remap_buffer) =
            Self::create_flags_buffer(device.clone(), groups_count, points_count, points_remap)?;

        let params = GpuVisitedFlagsParamsBuffer { generation: 1 };
        params_staging_buffer.upload(&params, 0)?;

        let mut upload_context = gpu::Context::new(device.clone())?;
        upload_context.clear_buffer(visited_flags_buffer.clone())?;
        upload_context.copy_gpu_buffer(
            params_staging_buffer.clone(),
            params_buffer.clone(),
            0,
            0,
            std::mem::size_of::<GpuVisitedFlagsParamsBuffer>(),
        )?;
        upload_context.run()?;
        upload_context.wait_finish(GPU_TIMEOUT)?;

        let mut descriptor_set_layout_builder = gpu::DescriptorSetLayout::builder()
            .add_uniform_buffer(0)
            .add_storage_buffer(1);
        if remap_buffer.is_some() {
            descriptor_set_layout_builder = descriptor_set_layout_builder.add_storage_buffer(2);
        }
        let descriptor_set_layout = descriptor_set_layout_builder.build(device.clone())?;

        let mut descriptor_set_builder = gpu::DescriptorSet::builder(descriptor_set_layout.clone())
            .add_uniform_buffer(0, params_buffer.clone())
            .add_storage_buffer(1, visited_flags_buffer.clone());
        if let Some(remap_buffer) = remap_buffer.clone() {
            descriptor_set_builder = descriptor_set_builder.add_storage_buffer(2, remap_buffer);
        }
        let descriptor_set = descriptor_set_builder.build()?;

        Ok(Self {
            params,
            params_buffer,
            params_staging_buffer,
            visited_flags_buffer,
            descriptor_set_layout,
            descriptor_set,
            capacity: points_count,
            remap: remap_buffer.is_some(),
        })
    }

    pub fn clear(&mut self, gpu_context: &mut gpu::Context) -> OperationResult<()> {
        if self.params.generation == 255 {
            self.params.generation = 1;
            gpu_context.clear_buffer(self.visited_flags_buffer.clone())?;
        } else {
            self.params.generation += 1;
        }

        self.params_staging_buffer.upload(&self.params, 0)?;
        gpu_context.copy_gpu_buffer(
            self.params_staging_buffer.clone(),
            self.params_buffer.clone(),
            0,
            0,
            self.params_buffer.size(),
        )?;
        Ok(())
    }

    pub fn descriptor_set_layout(&self) -> Arc<gpu::DescriptorSetLayout> {
        self.descriptor_set_layout.clone()
    }

    pub fn descriptor_set(&self) -> Arc<gpu::DescriptorSet> {
        self.descriptor_set.clone()
    }

    fn create_flags_buffer(
        device: Arc<gpu::Device>,
        groups_count: usize,
        points_count: usize,
        points_remap: &[PointOffsetType],
    ) -> OperationResult<(Arc<gpu::Buffer>, Option<Arc<gpu::Buffer>>)> {
        let visited_flags_buffer = gpu::Buffer::new(
            device.clone(),
            "Visited flags buffer",
            gpu::BufferType::Storage,
            groups_count * points_count * std::mem::size_of::<u8>(),
        )?;
        Ok((visited_flags_buffer, None))
    }
}
