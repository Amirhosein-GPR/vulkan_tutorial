use crate::app::AppData;
use crate::error::ApplicationError;
use ash::{extensions, vk, Entry, Instance};
use log::{debug, error, info, trace, warn};
use std::ffi::{c_void, CStr, CString};
use winit::window::Window;

// This version number is important for platforms that not fully conform to the Vulkan API specificaiton.
const PORTABILITY_MACOS_VERSION: u32 = vk::make_api_version(0, 1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER_NAME: &CStr =
    unsafe { &CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

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
