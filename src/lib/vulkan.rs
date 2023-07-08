use crate::app::AppData;
use crate::error::{AppError, SuitabilityError};
use ash::{extensions, vk, Device, Entry, Instance};
use log::{error, info, trace, warn};
use nalgebra::{Matrix4, Vector2, Vector3};
use png::{ColorType, Decoder};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_void, CStr, CString};
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::{mem, ptr};
use tobj::{self, LoadOptions};
use winit::platform::x11::WindowExtX11;
use winit::window::Window;

// This version number is important for platforms that not fully conform to the Vulkan API specificaiton.
const PORTABILITY_MACOS_VERSION: u32 = vk::make_api_version(0, 1, 3, 216);
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER_NAME: &CStr =
    unsafe { &CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };
const DEVICE_EXTENSIONS_NAMES: &[&CStr] = &[extensions::khr::Swapchain::name()];
pub const MAX_FRAMES_IN_FLIGHT: u8 = 2;

struct QueueFamilyIndices {
    graphics_queue_family: u32,
    present_queue_family: u32,
}

impl QueueFamilyIndices {
    // This function checks the given physical device for required queue families.
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        app_data: &mut AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, AppError> {
        let queue_family_properties =
            instance.get_physical_device_queue_family_properties(physical_device);

        // Checking for a graphics queue family.
        let graphics_queue_family = queue_family_properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        // Checking for a present queue family.
        let mut present_queue_family = None;
        let surface_loader = extensions::khr::Surface::new(entry, instance);
        for (index, _queue_family_properties) in queue_family_properties.into_iter().enumerate() {
            if surface_loader.get_physical_device_surface_support(
                physical_device,
                index as u32,
                app_data.surface,
            )? {
                present_queue_family = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics_queue_family), Some(present_queue_family)) =
            (graphics_queue_family, present_queue_family)
        {
            Ok(Self {
                graphics_queue_family,
                present_queue_family,
            })
        } else {
            Err(SuitabilityError(
                "Missing required queue families.".to_string(),
            ))?
        }
    }
}

struct SwapchainSupport {
    surface_capabilites: vk::SurfaceCapabilitiesKHR,
    surface_formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    // Gets physical device surface properties.
    unsafe fn get(
        entry: &Entry,
        instance: &Instance,
        app_data: &mut AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, AppError> {
        let surface_loader = extensions::khr::Surface::new(entry, instance);

        Ok(Self {
            surface_capabilites: surface_loader
                .get_physical_device_surface_capabilities(physical_device, app_data.surface)?,
            surface_formats: surface_loader
                .get_physical_device_surface_formats(physical_device, app_data.surface)?,
            present_modes: surface_loader
                .get_physical_device_surface_present_modes(physical_device, app_data.surface)?,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    position: Vector3<f32>,
    color: Vector3<f32>,
    texture_coordinate: Vector2<f32>,
}

impl Vertex {
    pub fn new(
        position: Vector3<f32>,
        color: Vector3<f32>,
        texture_coordinate: Vector2<f32>,
    ) -> Self {
        Self {
            position,
            color,
            texture_coordinate,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let position = vk::VertexInputAttributeDescription::builder()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(mem::size_of::<Vector3<f32>>() as u32)
            .build();
        let texture_coordinate = vk::VertexInputAttributeDescription::builder()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((mem::size_of::<Vector3<f32>>() + mem::size_of::<Vector3<f32>>()) as u32)
            .build();

        [position, color, texture_coordinate]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.position == other.position
            && self.color == other.color
            && self.texture_coordinate == other.texture_coordinate
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.position[0].to_bits().hash(state);
        self.position[1].to_bits().hash(state);
        self.position[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.texture_coordinate[0].to_bits().hash(state);
        self.texture_coordinate[1].to_bits().hash(state);
    }
}

#[repr(C)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub project: Matrix4<f32>,
}

// Creates a Vulkan instance.
pub unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    app_data: &mut AppData,
) -> Result<Instance, AppError> {
    let application_name = CString::new("Vulkan tutorial").unwrap();
    let engine_name = CString::new("Vulkan engine").unwrap();
    let application_info = vk::ApplicationInfo::builder()
        .application_name(&application_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    let mut instance_layers_names = Vec::new();
    // These extensions are needed for surface creation. Surface is our link to window for preseneting images on it.
    let mut instance_extensions_names = vec![
        extensions::khr::Surface::name().as_ptr(),
        extensions::khr::XlibSurface::name().as_ptr(),
    ];

    if VALIDATION_ENABLED {
        // Enumerating instance layer properties. First getting them in [i8] format, then creating a slice of CStr from them.
        let available_instance_layer_names = entry
            .enumerate_instance_layer_properties()?
            .into_iter()
            .map(|l| l.layer_name)
            .collect::<Vec<_>>();
        let available_instance_layer_names = available_instance_layer_names
            .iter()
            .map(|l| CStr::from_ptr(l.as_ptr()))
            .collect::<Vec<_>>();

        if !available_instance_layer_names.contains(&VALIDATION_LAYER_NAME) {
            return Err(AppError::EngineError(
                "Validation layer requested but not supported.".to_string(),
            ));
        }

        instance_layers_names.push(VALIDATION_LAYER_NAME.as_ptr());
        instance_extensions_names.push(extensions::ext::DebugUtils::name().as_ptr());
    }

    /* Starting from Vulkan version 1.3.216, platforms that not fully conform to the Vulkan API specification should enable
    these extensions and pass them with this flag to instance create info struct. */
    let instance_create_flags = if cfg!(target_os = "macos")
        && entry.try_enumerate_instance_version()?.unwrap() >= PORTABILITY_MACOS_VERSION
    {
        info!("Enabling extensions for macOS portability.");

        instance_extensions_names
            .push(extensions::khr::GetPhysicalDeviceProperties2::name().as_ptr());
        instance_extensions_names.push(vk::KhrPortabilityEnumerationFn::name().as_ptr());

        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let mut instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&instance_layers_names)
        .enabled_extension_names(&instance_extensions_names)
        .flags(instance_create_flags);

    // Creating debug utils messenge which is used to enable validation layers.
    let instance;
    if VALIDATION_ENABLED {
        let mut debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_callback));

        instance_create_info =
            instance_create_info.push_next(&mut debug_utils_messenger_create_info);

        instance = entry.create_instance(&instance_create_info, None)?;

        app_data.debug_utils_messenger = extensions::ext::DebugUtils::new(entry, &instance)
            .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)?;
    } else {
        instance = entry.create_instance(&instance_create_info, None)?;
    }

    Ok(instance)
}

extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*callback_data).p_message) }.to_string_lossy();

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => error!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!("({:?}) {}", message_type, message),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => info!("({:?}) {}", message_type, message),
        _ => {
            trace!("({:?}), {}", message_type, message)
        }
    }

    vk::FALSE
}

// This function picks the required physical device.
pub unsafe fn pick_physical_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    for physical_device in instance.enumerate_physical_devices()? {
        let physical_device_name = CStr::from_ptr(
            instance
                .get_physical_device_properties(physical_device)
                .device_name
                .as_ptr(),
        );

        if let Err(error) = check_physical_device(entry, instance, app_data, physical_device) {
            warn!(
                "Skipping physical device ('{:?}'): {}",
                physical_device_name, error
            );
        } else {
            info!("Selected physical device ('{:?}').", physical_device_name);
            app_data.physical_device = physical_device;
            return Ok(());
        }
    }

    Err(SuitabilityError(
        "Failed to find suitable physical device.".to_string(),
    ))?
}

// This function checks the given physical device for required properties and features.
unsafe fn check_physical_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<(), AppError> {
    let physical_device_properties = instance.get_physical_device_properties(physical_device);
    if physical_device_properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(SuitabilityError(
            "Only discrete GPUs are supported.".to_string(),
        ))?;
    }

    let physical_device_features = instance.get_physical_device_features(physical_device);
    if physical_device_features.sampler_anisotropy != vk::TRUE {
        return Err(SuitabilityError("Missing sampler anisotropy.".to_string()))?;
    }

    QueueFamilyIndices::get(entry, instance, app_data, physical_device)?;

    check_physical_device_extensions(instance, physical_device)?;

    // If there isn't any surface formats or present modes available, then there's no sufficient support for swapchain in current physical device.
    let swapchain_support = SwapchainSupport::get(entry, instance, app_data, physical_device)?;
    if swapchain_support.surface_formats.is_empty() || swapchain_support.present_modes.is_empty() {
        return Err(SuitabilityError(
            "Insufficient swapchain support.".to_string(),
        ))?;
    }

    Ok(())
}

// This function creates a logical device and retrieves queue handles that are automatically created with the logical device.
pub unsafe fn create_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<Device, AppError> {
    let queue_family_indices =
        QueueFamilyIndices::get(entry, instance, app_data, app_data.physical_device)?;

    // Using hash set to avoid possible duplicated queue family indices.
    let mut unique_indices = HashSet::new();
    unique_indices.insert(queue_family_indices.graphics_queue_family);
    unique_indices.insert(queue_family_indices.present_queue_family);

    let queue_priorities = [1.0];
    let device_queue_create_infos = unique_indices
        .into_iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(i)
                .queue_priorities(&queue_priorities)
                .build()
        })
        .collect::<Vec<_>>();

    let mut device_layers_names = Vec::new();
    let mut device_extensions_names = DEVICE_EXTENSIONS_NAMES
        .into_iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();
    let device_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);
    if VALIDATION_ENABLED {
        device_layers_names.push(VALIDATION_LAYER_NAME.as_ptr());
    }

    /* Starting from Vulkan version 1.3.216, platforms that not fully conform to the Vulkan API specification should enable
    this extension and pass it to device create info struct. */
    if cfg!(target_os = "macos")
        && entry.try_enumerate_instance_version()?.unwrap() >= PORTABILITY_MACOS_VERSION
    {
        device_extensions_names.push(vk::KhrPortabilitySubsetFn::name().as_ptr());
    }

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&device_queue_create_infos)
        .enabled_layer_names(&device_layers_names)
        .enabled_extension_names(&device_extensions_names)
        .enabled_features(&device_features);

    let device = instance.create_device(app_data.physical_device, &device_create_info, None)?;

    app_data.graphics_queue =
        device.get_device_queue(queue_family_indices.graphics_queue_family, 0);
    app_data.present_queue = device.get_device_queue(queue_family_indices.present_queue_family, 0);

    Ok(device)
}

// Creates a surface with the help of winit and vulkan extensions.
pub unsafe fn create_surface(
    entry: &Entry,
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR, AppError> {
    let xlib_surface_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .dpy(window.xlib_display().unwrap() as *mut vk::Display)
        .window(window.xlib_window().unwrap());

    let surface_loader = extensions::khr::XlibSurface::new(entry, instance);
    let surface = surface_loader.create_xlib_surface(&xlib_surface_create_info, None)?;

    Ok(surface)
}

// Checks physical device to support logical device extensions.
unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(), AppError> {
    let available_device_extensions_names = instance
        .enumerate_device_extension_properties(physical_device)?
        .into_iter()
        .map(|e| e.extension_name)
        .collect::<Vec<_>>();
    let available_device_extensions_names = available_device_extensions_names
        .iter()
        .map(|e| CStr::from_ptr(e.as_ptr()))
        .collect::<Vec<_>>();

    if DEVICE_EXTENSIONS_NAMES
        .iter()
        .all(|e| available_device_extensions_names.contains(e))
    {
        Ok(())
    } else {
        Err(SuitabilityError(
            "Missing required device extensions.".to_string(),
        ))?
    }
}

// Picks one of many possible surface formats.
unsafe fn get_swapchain_surface_format(
    surface_fomrats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    surface_fomrats
        .into_iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| surface_fomrats[0])
}

// Picks one of many possible surface present modes.
unsafe fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .into_iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or_else(|| vk::PresentModeKHR::FIFO)
}

// Gets surface extent from surface capabilites or creates a new one if we can decide on it's size.
unsafe fn get_swapchain_extent(
    window: &Window,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    /* u32::MAX is the maximum u32 number value. Here it has a meaning. If current extent width is equal to u32::MAX,
    it means we can pick a resolution that best matches the window within min image extent and max image extent.*/
    if surface_capabilities.current_extent.width != u32::MAX {
        surface_capabilities.current_extent
    } else {
        let window_size = window.inner_size();
        let clamp = |min: u32, max: u32, value: u32| min.max(max.min(value));
        vk::Extent2D::builder()
            .width(clamp(
                surface_capabilities.min_image_extent.width,
                surface_capabilities.max_image_extent.width,
                window_size.width,
            ))
            .height(clamp(
                surface_capabilities.min_image_extent.height,
                surface_capabilities.max_image_extent.height,
                window_size.height,
            ))
            .build()
    }
}

// Creates swapchain and retrieves handles to images in it and store them in app data.
pub unsafe fn create_swapchain(
    entry: &Entry,
    window: &Window,
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let queue_family_indices =
        QueueFamilyIndices::get(entry, instance, app_data, app_data.physical_device)?;
    let swapchain_support =
        SwapchainSupport::get(entry, instance, app_data, app_data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&swapchain_support.surface_formats);
    let present_mode = get_swapchain_present_mode(&swapchain_support.present_modes);
    let extent = get_swapchain_extent(window, swapchain_support.surface_capabilites);

    let mut image_count = swapchain_support.surface_capabilites.min_image_count + 1;
    // '0' for max image count is a special value which means there is no maximum for image count.
    if swapchain_support.surface_capabilites.max_image_count != 0
        && image_count > swapchain_support.surface_capabilites.max_image_count
    {
        image_count = swapchain_support.surface_capabilites.max_image_count;
    }

    let mut seperate_queue_family_indices = Vec::new();
    // Concurrent image sharing mode is used for at least 2 different queue families.
    let image_sharing_mode = if queue_family_indices.graphics_queue_family
        != queue_family_indices.present_queue_family
    {
        seperate_queue_family_indices.push(queue_family_indices.graphics_queue_family);
        seperate_queue_family_indices.push(queue_family_indices.present_queue_family);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(app_data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&seperate_queue_family_indices)
        .pre_transform(swapchain_support.surface_capabilites.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain_loader = extensions::khr::Swapchain::new(instance, device);
    app_data.swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None)?;
    app_data.swapchain_images = swapchain_loader.get_swapchain_images(app_data.swapchain)?;
    app_data.swapchain_format = surface_format.format;
    app_data.swapchain_extent = extent;

    Ok(())
}

// Creates image views, which are needed to work with images in the swapchain.
pub unsafe fn create_swapchain_image_views(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    app_data.swapchain_image_views = app_data
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                app_data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

pub unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let color_attachment_descriptions = vk::AttachmentDescription::builder()
        .format(app_data.swapchain_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();
    let color_attachment_refrences = [vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()];

    let depth_attachment_descriptions = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, app_data)?)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();
    let depth_attachment_refrences = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass_descriptions = [vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refrences)
        .depth_stencil_attachment(&depth_attachment_refrences)
        .build()];

    let subpass_dependencies = [vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .build()];

    let attachment_descriptions = [color_attachment_descriptions, depth_attachment_descriptions];
    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descriptions)
        .subpasses(&subpass_descriptions)
        .dependencies(&subpass_dependencies);

    app_data.render_pass = device.create_render_pass(&render_pass_create_info, None)?;

    Ok(())
}

// Creates pipeline by specifying it's required states and stages.
pub unsafe fn create_pipeline(device: &Device, app_data: &mut AppData) -> Result<(), AppError> {
    let vertex_shader_spv = fs::read("shaders/vert.spv")?;
    let fragment_shader_spv = fs::read("shaders/frag.spv")?;

    let vertex_shader_module = create_shader_module(device, &vertex_shader_spv)?;
    let fragment_shader_module = create_shader_module(device, &fragment_shader_spv)?;

    let vertex_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
        .build();
    let fragment_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"))
        .build();

    let vertex_binding_descriptions = [Vertex::binding_description()];
    let vertex_attribute_descriptions = Vertex::attribute_descriptions();
    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&vertex_binding_descriptions)
        .vertex_attribute_descriptions(&vertex_attribute_descriptions);

    let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewports = [vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(app_data.swapchain_extent.width as f32)
        .height(app_data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)
        .build()];
    let scissor_rectangles = [vk::Rect2D::builder()
        .offset(vk::Offset2D::builder().x(0).y(0).build())
        .extent(app_data.swapchain_extent)
        .build()];

    let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissor_rectangles);

    let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stenci_state_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false);

    // This color blending configuration is per attached framebuffer
    let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()];

    // This color blending configuration is global
    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachment_states)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // If we want to use dynamic states, we should mention them here.
    // let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
    // let dynamic_state_create_info =
    //     vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    // Pipeline layout is used for creating uniform values which are used in shaders and their values can be changed at drawing times.
    let descriptor_set_layouts = [app_data.descriptor_set_layout];
    let pipeline_layout_create_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);

    app_data.pipeline_layout = device.create_pipeline_layout(&pipeline_layout_create_info, None)?;

    let shader_stage_create_infos = [
        vertex_shader_stage_create_info,
        fragment_shader_stage_create_info,
    ];

    let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stage_create_infos)
        .vertex_input_state(&vertex_input_state_create_info)
        .input_assembly_state(&input_assembly_state_create_info)
        .viewport_state(&viewport_state_create_info)
        .rasterization_state(&rasterization_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .depth_stencil_state(&depth_stenci_state_create_info)
        .color_blend_state(&color_blend_state_create_info)
        // .dynamic_state(&dynamic_state_create_info)
        .layout(app_data.pipeline_layout)
        .render_pass(app_data.render_pass)
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1)
        .build()];

    app_data.pipeline = device.create_graphics_pipelines(
        vk::PipelineCache::null(),
        &pipeline_create_infos,
        None,
    )?[0];

    device.destroy_shader_module(vertex_shader_module, None);
    device.destroy_shader_module(fragment_shader_module, None);

    Ok(())
}

// Creates shader module which is needed to link each shader to pipeline.
unsafe fn create_shader_module(
    device: &Device,
    bytecode: &[u8],
) -> Result<vk::ShaderModule, AppError> {
    // Shader module create info needs a u32 slice but what we read is a u8 slice. So we align our u8 slice to a u32 slice.
    let (prefix, code, suffix) = bytecode.align_to::<u32>();
    // We also check if our slice is aligned properly which means there shouldn't be any prefix or suffix for the aligned slice.
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err(AppError::EngineError(
            "Shader bytecode is not properly aligned.".to_string(),
        ));
    }

    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&code);

    Ok(device.create_shader_module(&shader_module_create_info, None)?)
}

pub unsafe fn create_framebuffers(device: &Device, app_data: &mut AppData) -> Result<(), AppError> {
    app_data.framebuffers = app_data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = [*i, app_data.depth_image_view];
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(app_data.render_pass)
                .attachments(&attachments)
                .width(app_data.swapchain_extent.width)
                .height(app_data.swapchain_extent.height)
                .layers(1);
            device.create_framebuffer(&framebuffer_create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

// Creates command pool which is required to allocate command buffers.
pub unsafe fn create_command_pool(
    entry: &Entry,
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let queue_family_indices =
        QueueFamilyIndices::get(entry, instance, app_data, app_data.physical_device)?;
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(queue_family_indices.graphics_queue_family);

    app_data.command_pool = device.create_command_pool(&command_pool_create_info, None)?;

    Ok(())
}

// Creates command buffers and records some commands into them.
pub unsafe fn create_command_buffers(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(app_data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(app_data.framebuffers.len() as u32);

    app_data.command_buffers = device.allocate_command_buffers(&command_buffer_allocate_info)?;

    // Only relevant for secondary command buffers.
    let command_buffer_inheritance_info = vk::CommandBufferInheritanceInfo::builder();
    let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::empty())
        .inheritance_info(&command_buffer_inheritance_info);

    let render_area = vk::Rect2D::builder()
        .offset(vk::Offset2D::default())
        .extent(app_data.swapchain_extent)
        .build();

    let clear_color_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];

    for (i, command_buffer) in app_data.command_buffers.iter().enumerate() {
        device.begin_command_buffer(*command_buffer, &command_buffer_begin_info)?;

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(app_data.render_pass)
            .framebuffer(app_data.framebuffers[i])
            .render_area(render_area)
            .clear_values(&clear_color_values);

        device.cmd_begin_render_pass(
            *command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app_data.pipeline,
        );
        device.cmd_bind_vertex_buffers(*command_buffer, 0, &[app_data.vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(
            *command_buffer,
            app_data.index_buffer,
            0,
            vk::IndexType::UINT32,
        );
        device.cmd_bind_descriptor_sets(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app_data.pipeline_layout,
            0,
            &[app_data.descriptor_sets[i]],
            &[],
        );
        device.cmd_draw_indexed(*command_buffer, app_data.indices.len() as u32, 1, 0, 0, 0);
        device.cmd_end_render_pass(*command_buffer);

        device.end_command_buffer(*command_buffer)?;
    }

    Ok(())
}

pub unsafe fn create_sync_objects(device: &Device, app_data: &mut AppData) -> Result<(), AppError> {
    let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
    let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        app_data
            .image_available_semaphores
            .push(device.create_semaphore(&semaphore_create_info, None)?);
        app_data
            .render_finished_semaphores
            .push(device.create_semaphore(&semaphore_create_info, None)?);
        app_data
            .in_flight_frame_fences
            .push(device.create_fence(&fence_create_info, None)?);
    }

    app_data.in_flight_image_fences = app_data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

unsafe fn get_memory_type_index(
    instance: &Instance,
    app_data: &mut AppData,
    memory_property_flags: vk::MemoryPropertyFlags,
    memory_requirements: vk::MemoryRequirements,
) -> Result<u32, AppError> {
    let memory_properties =
        instance.get_physical_device_memory_properties(app_data.physical_device);

    (0..memory_properties.memory_type_count)
        .find(|i| {
            let suitable = memory_requirements.memory_type_bits & (1 << i) != 0;
            let memory_type = memory_properties.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(memory_property_flags)
        })
        .ok_or(AppError::EngineError(
            "Failed to find suitable memory type.".to_string(),
        ))
}

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
    size: vk::DeviceSize,
    buffer_usage_flags: vk::BufferUsageFlags,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), AppError> {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(buffer_usage_flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_create_info, None)?;

    let memory_requirements = device.get_buffer_memory_requirements(buffer);

    let memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            app_data,
            memory_property_flags,
            memory_requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_allocate_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn copy_buffer(
    device: &Device,
    app_data: &mut AppData,
    source_buffer: vk::Buffer,
    destination_buffer: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<(), AppError> {
    let command_buffer = begin_single_time_commands(device, app_data)?;

    let buffer_copy_regions = vk::BufferCopy::builder().size(size).build();

    device.cmd_copy_buffer(
        command_buffer,
        source_buffer,
        destination_buffer,
        &[buffer_copy_regions],
    );

    end_single_time_commands(device, app_data, command_buffer)?;

    Ok(())
}

pub unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let size = (mem::size_of::<Vertex>() * app_data.vertices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        app_data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    ptr::copy_nonoverlapping(
        app_data.vertices.as_ptr(),
        memory.cast(),
        app_data.vertices.len(),
    );

    device.unmap_memory(staging_buffer_memory);

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        app_data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    app_data.vertex_buffer = vertex_buffer;
    app_data.vertex_buffer_memory = vertex_buffer_memory;

    copy_buffer(device, app_data, staging_buffer, vertex_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

pub unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let size = (mem::size_of::<u32>() * app_data.indices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        app_data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    ptr::copy_nonoverlapping(
        app_data.indices.as_ptr(),
        memory.cast(),
        app_data.indices.len(),
    );

    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        app_data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    app_data.index_buffer = index_buffer;
    app_data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, app_data, staging_buffer, index_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

pub unsafe fn create_descriptor_set_layout(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let descriptor_set_layout_bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build(),
    ];

    let descriptor_set_layout_create_info =
        vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_layout_bindings);

    app_data.descriptor_set_layout =
        device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?;

    Ok(())
}

pub unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    app_data.uniform_buffers.clear();
    app_data.uniform_buffers_memories.clear();

    for _ in 0..app_data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            app_data,
            mem::size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        app_data.uniform_buffers.push(uniform_buffer);
        app_data
            .uniform_buffers_memories
            .push(uniform_buffer_memory);
    }

    Ok(())
}

pub unsafe fn create_descriptor_pool(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let descriptor_pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(app_data.swapchain_images.len() as u32)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(app_data.swapchain_images.len() as u32)
            .build(),
    ];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(app_data.swapchain_images.len() as u32)
        .pool_sizes(&descriptor_pool_sizes);

    app_data.descriptor_pool = device.create_descriptor_pool(&descriptor_pool_create_info, None)?;

    Ok(())
}

pub unsafe fn create_descriptor_sets(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let descriptor_set_layouts =
        vec![app_data.descriptor_set_layout; app_data.swapchain_images.len()];
    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(app_data.descriptor_pool)
        .set_layouts(&descriptor_set_layouts);

    app_data.descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_allocate_info)?;

    for i in 0..app_data.swapchain_images.len() {
        let descriptor_buffer_infos = [vk::DescriptorBufferInfo::builder()
            .buffer(app_data.uniform_buffers[i])
            .offset(0)
            .range(mem::size_of::<UniformBufferObject>() as u64)
            .build()];

        let descriptor_image_infos = [vk::DescriptorImageInfo::builder()
            .sampler(app_data.texture_sampler)
            .image_view(app_data.texture_image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build()];

        let write_descriptor_sets = [
            vk::WriteDescriptorSet::builder()
                .dst_set(app_data.descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&descriptor_buffer_infos)
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(app_data.descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&descriptor_image_infos)
                .build(),
        ];

        device.update_descriptor_sets(&write_descriptor_sets, &[]);
    }

    Ok(())
}

pub unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let image_file = File::open("resources/viking_room.png")?;

    let decoder = Decoder::new(image_file);
    let mut reader = decoder.read_info()?;

    let size = reader.info().raw_bytes() as u64;

    let mut pixels = vec![0; size as usize];
    reader.next_frame(&mut pixels)?;

    let (width, height) = reader.info().size();
    app_data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    if width != 1024 || height != 1024 || reader.info().color_type != ColorType::Rgba {
        return Err(SuitabilityError("Invalid texture image.".to_string()))?;
    }

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        app_data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    ptr::copy_nonoverlapping(pixels.as_ptr(), memory.cast(), pixels.len());

    device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        app_data,
        width,
        height,
        app_data.mip_levels,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    app_data.texture_image = texture_image;
    app_data.texture_image_memory = texture_image_memory;

    transition_image_layout(
        device,
        app_data,
        app_data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        app_data.mip_levels,
    )?;

    copy_buffer_to_image(
        device,
        app_data,
        staging_buffer,
        app_data.texture_image,
        width,
        height,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    generate_mipmaps(
        instance,
        device,
        app_data,
        app_data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        app_data.mip_levels,
    )?;

    Ok(())
}

unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    format: vk::Format,
    image_tiling: vk::ImageTiling,
    image_usage_flags: vk::ImageUsageFlags,
    memory_property_flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory), AppError> {
    let image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(image_tiling)
        .usage(image_usage_flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = device.create_image(&image_create_info, None)?;

    let image_memory_requirements = device.get_image_memory_requirements(image);

    let memory_allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(image_memory_requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            app_data,
            memory_property_flags,
            image_memory_requirements,
        )?);

    let image_memory = device.allocate_memory(&memory_allocate_info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn begin_single_time_commands(
    device: &Device,
    app_data: &mut AppData,
) -> Result<vk::CommandBuffer, AppError> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(app_data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&command_buffer_allocate_info)?[0];

    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &command_buffer_begin_info)?;

    Ok(command_buffer)
}

unsafe fn end_single_time_commands(
    device: &Device,
    app_data: &mut AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<(), AppError> {
    device.end_command_buffer(command_buffer)?;

    let command_buffers = [command_buffer];
    let submit_infos = [vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .build()];

    device.queue_submit(app_data.graphics_queue, &submit_infos, vk::Fence::null())?;

    device.queue_wait_idle(app_data.graphics_queue)?;

    device.free_command_buffers(app_data.command_pool, &command_buffers);

    Ok(())
}

unsafe fn transition_image_layout(
    device: &Device,
    app_data: &mut AppData,
    image: vk::Image,
    format: vk::Format,
    old_image_layout: vk::ImageLayout,
    new_image_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<(), AppError> {
    let image_aspect_flags =
        if new_image_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
            match format {
                vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                }
                _ => vk::ImageAspectFlags::DEPTH,
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_image_layout, new_image_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => {
                return Err(SuitabilityError(
                    "Unsupported image layout transition!".to_string(),
                ))?
            }
        };

    let command_buffer = begin_single_time_commands(device, app_data)?;

    let image_subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(image_aspect_flags)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let image_memory_barriers = [vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_image_layout)
        .new_layout(new_image_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(image_subresource_range)
        .build()];

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &image_memory_barriers,
    );

    end_single_time_commands(device, app_data, command_buffer)?;

    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    app_data: &mut AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<(), AppError> {
    let command_buffer = begin_single_time_commands(device, app_data)?;

    let image_subresource_layers = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let buffer_image_copies = [vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(image_subresource_layers)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .build()];

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &buffer_image_copies,
    );

    end_single_time_commands(device, app_data, command_buffer)?;

    Ok(())
}

pub unsafe fn create_texture_image_view(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    app_data.texture_image_view = create_image_view(
        device,
        app_data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        app_data.mip_levels,
    )?;

    Ok(())
}

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    image_aspect_flags: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView, AppError> {
    let image_subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(image_aspect_flags)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let image_view_create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(image_subresource_range);

    Ok(device.create_image_view(&image_view_create_info, None)?)
}

pub unsafe fn create_texture_sampler(
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let sampler_create_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .mip_lod_bias(0.0)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .min_lod(0.0)
        .max_lod(app_data.mip_levels as f32)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false);

    app_data.texture_sampler = device.create_sampler(&sampler_create_info, None)?;

    Ok(())
}

pub unsafe fn create_depth_objects(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
) -> Result<(), AppError> {
    let format = get_depth_format(instance, app_data)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        app_data,
        app_data.swapchain_extent.width,
        app_data.swapchain_extent.height,
        1,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    app_data.depth_image = depth_image;
    app_data.depth_image_memory = depth_image_memory;

    app_data.depth_image_view = create_image_view(
        device,
        app_data.depth_image,
        format,
        vk::ImageAspectFlags::DEPTH,
        1,
    )?;

    transition_image_layout(
        device,
        app_data,
        app_data.depth_image,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        1,
    )?;

    Ok(())
}

unsafe fn get_supported_format(
    instance: &Instance,
    app_data: &mut AppData,
    format_candidates: &[vk::Format],
    image_tiling: vk::ImageTiling,
    format_feature_flags: vk::FormatFeatureFlags,
) -> Result<vk::Format, AppError> {
    format_candidates
        .iter()
        .cloned()
        .find(|f| {
            let format_properties =
                instance.get_physical_device_format_properties(app_data.physical_device, *f);

            match image_tiling {
                vk::ImageTiling::LINEAR => format_properties
                    .linear_tiling_features
                    .contains(format_feature_flags),
                vk::ImageTiling::OPTIMAL => format_properties
                    .optimal_tiling_features
                    .contains(format_feature_flags),
                _ => false,
            }
        })
        .ok_or(SuitabilityError("Failed to find supported format!".to_string()).into())
}

unsafe fn get_depth_format(
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<vk::Format, AppError> {
    let format_candidates = [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        app_data,
        &format_candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

pub unsafe fn load_model(app_data: &mut AppData) -> Result<(), AppError> {
    let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &LoadOptions {
            triangulate: true,
            ..Default::default()
        },
        |_| Ok(Default::default()),
    )?;

    let mut unique_vertices = HashMap::new();

    for model in models {
        for index in model.mesh.indices {
            let positions_offset = (3 * index) as usize;
            let texture_coordinate_offset = (2 * index) as usize;

            let vertex = Vertex {
                position: Vector3::new(
                    model.mesh.positions[positions_offset],
                    model.mesh.positions[positions_offset + 1],
                    model.mesh.positions[positions_offset + 2],
                ),
                color: Vector3::new(1.0, 1.0, 1.0),
                texture_coordinate: Vector2::new(
                    model.mesh.texcoords[texture_coordinate_offset],
                    1.0 - model.mesh.texcoords[texture_coordinate_offset + 1],
                ),
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                app_data.indices.push(*index as u32);
            } else {
                let index = app_data.vertices.len();
                unique_vertices.insert(vertex, index);
                app_data.vertices.push(vertex);
                app_data.indices.push(index as u32);
            }
        }
    }

    Ok(())
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    app_data: &mut AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<(), AppError> {
    if !instance
        .get_physical_device_format_properties(app_data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(SuitabilityError(
            "Texture image format does not support linear blitting!".to_string(),
        ))?;
    }

    let command_buffer = begin_single_time_commands(device, app_data)?;

    let image_subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    let mut image_memory_barrier = vk::ImageMemoryBarrier {
        image,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        subresource_range: image_subresource_range,
        ..Default::default()
    };

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        image_memory_barrier.subresource_range.base_mip_level = i - 1;
        image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        image_memory_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        image_memory_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[image_memory_barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let image_blit = vk::ImageBlit::builder()
            .src_subresource(src_subresource)
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .build();

        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[image_blit],
            vk::Filter::LINEAR,
        );

        image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        image_memory_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[image_memory_barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    image_memory_barrier.subresource_range.base_mip_level = mip_levels - 1;
    image_memory_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    image_memory_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    image_memory_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    image_memory_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[image_memory_barrier],
    );

    end_single_time_commands(device, app_data, command_buffer)?;

    Ok(())
}
