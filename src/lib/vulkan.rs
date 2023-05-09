use crate::app::AppData;
use crate::error::{ApplicationError, SuitabilityError};
use ash::{extensions, vk, Device, Entry, Instance};
use log::{debug, error, info, trace, warn};
use std::ffi::{c_void, CStr, CString};
use winit::window::Window;

// This version number is important for platforms that not fully conform to the Vulkan API specificaiton.
const PORTABILITY_MACOS_VERSION: u32 = vk::make_api_version(0, 1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER_NAME: &CStr =
    unsafe { &CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

struct QueueFamilyIndices {
    graphics_queue_family: u32,
}

impl QueueFamilyIndices {
    // This function checks the given physical device for required queue families.
    unsafe fn get(
        instance: &Instance,
        app_data: &mut AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self, ApplicationError> {
        let queue_family_properties =
            instance.get_physical_device_queue_family_properties(physical_device);

        let graphics_queue_family = queue_family_properties
            .into_iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        if let Some(graphics_queue_family) = graphics_queue_family {
            Ok(Self {
                graphics_queue_family,
            })
        } else {
            Err(SuitabilityError(
                "Missing required queue families.".to_string(),
            ))?
        }
    }
}

// Creates a Vulkan instance.
pub unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    app_data: &mut AppData,
) -> Result<Instance, ApplicationError> {
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
            return Err(ApplicationError::EngineError(
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
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<(), ApplicationError> {
    for physical_device in instance.enumerate_physical_devices()? {
        let physical_device_name = CStr::from_ptr(
            instance
                .get_physical_device_properties(physical_device)
                .device_name
                .as_ptr(),
        );

        if let Err(error) = check_physical_device(instance, app_data, physical_device) {
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
    instance: &Instance,
    app_data: &mut AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<(), ApplicationError> {
    let physical_device_properties = instance.get_physical_device_properties(physical_device);
    if physical_device_properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(SuitabilityError(
            "Only discrete GPUs are supported.".to_string(),
        ))?;
    }

    let physical_device_features = instance.get_physical_device_features(physical_device);
    if physical_device_features.geometry_shader != vk::TRUE {
        return Err(SuitabilityError(
            "Missing geometry shader support.".to_string(),
        ))?;
    }

    QueueFamilyIndices::get(instance, app_data, physical_device)?;

    Ok(())
}

// This function creates a logical device and retrieves queue handles that are automatically created with the logical device.
pub unsafe fn create_device(
    entry: &Entry,
    instance: &Instance,
    app_data: &mut AppData,
) -> Result<Device, ApplicationError> {
    let queue_family_indices =
        QueueFamilyIndices::get(instance, app_data, app_data.physical_device)?;

    let queue_prioriteis = [1.0];
    let device_queue_create_infos = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_indices.graphics_queue_family)
        .queue_priorities(&queue_prioriteis)
        .build()];

    let mut device_layers_names = Vec::new();
    let mut device_extensions_names = Vec::new();
    let device_features = vk::PhysicalDeviceFeatures::builder();
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

    Ok(device)
}
