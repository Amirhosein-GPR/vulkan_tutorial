## 1. Window and event loop creation
#### 1.1. Create event loop
#### 1.2. Create window
#### 1.3. Run event loop and manage needed states

## 2. Instance creation
#### 2.1. Create application info
#### 2.2. Create instance extensions names
#### 2.3. Create instance create flags
#### 2.4. Create instance create info
#### 2.5. Create Instance (After 3.5 If we want validation layers for Vulkan instance)

## 3. Enabling validation layers when we want
#### 3.1. Enumerate available instance layer names
#### 3.2. Check if 'VK_LAYER_KHRONOS_validation' is in the available names
#### 3.3. Add necessary layer and extension name for Vulkan instance.
#### 3.4. Create debug utils messenger create info
#### 3.5. Push previous struct to the instance create info struct
#### 3.6. Create debug utils messenger

## 4. Window surface creation
#### 4.1. Create preferred platform surface create info
#### 4.2. Create preferred platform surface loader
#### 4.3. Create surface with previos step surface loader

## 5. Physical device and queue families selection
#### 5.1. Enumerate physical devices
#### 5.2. Check for appropriate physical device
##### 5.2.1. Check for appropriate physical device properties
##### 5.2.2. Check for appropriate physical device features
##### 5.2.3. Check for appropriate physical device queue families
#### 5.3. If physical device passes all checks, store physical device and needed queue family indices in relevant structs

## 6. Logical device creation and acquiring queues
#### 6.1. Get required queue family indices
#### 6.2. Create device queue create info
#### 6.3. Create required list of device layers names, extensions names and features
#### 6.4. Create device create info
#### 6.5. Create device
#### 6.6. Get required queue handles and store them in an appropriate place

## 7. Swapchain creation and acquiring it's images
#### 7.1. Checks physical device to support logical device extensions, including swapchain extension
#### 7.2. Checks physical device and window for surface properties needed for creating a swapchain
#### 7.3. Create swapchain create info
#### 7.3. Create swapchain
#### 7.4. Get swapchain images handles and store them in an appropriate place
#### 7.5. Store swapchain format and swapchain extent in an appropriate place

## 8. Creating image views
#### 8.1. Create image view create info
##### 8.1.1 Create component mapping
##### 8.1.2 Create subresource range
##### 8.1.2 Create image view create info
#### 8.2. Create image views

## 9. Creating shader modules
#### 9.1. Write required vertex shader and fragment shader codes in seperate files
#### 9.2. Compile shader codes
#### 9.3. Read compiled shader codes (bytecodes) to the program
#### 9.4. Create shader stage create info for vertex and fragment shaders
#### 9.5. Destroy shader modules after graphics pipeline has been created

## 10. Configuring fixed functions
#### 10.1. Create vertex input state create info
#### 10.2. Create input assembly state create info
#### 10.3. Create viewport and scissor
#### 10.4. Create viewport state create info
#### 10.5. Create rasterization state create info
#### 10.6. Create multisample state create info
#### 10.7. Create depth stencil state create info
#### 10.8. Create color blend attachment state
#### 10.9. Create color blend state create info
#### 10.11. Create dynamic states
#### 10.12. Create dynamic state create info
#### 10.13. Create pipeline layout create info
#### 10.14. Create pipeline layout

## 11. Creating render passes
#### 11.1. Create attachment descriptions
#### 11.2. Create attachment refrences
#### 11.3. Create subpass descriptions
#### 11.4. Create render pass create info
#### 11.5. Create render pass


## 12. Creating graphics pipeline
#### 12.1. Create graphics pipeline create info
#### 12.2. Create graphics pipeline

## 13. Creating framebuffers
#### 13.1. Create framebuffer create infos
#### 13.2. Create framebuffers

## 14. Creating command buffers
#### 14.1. Create command pool create info
#### 14.2. Create command pool
#### 14.3. Create command buffer allocate info
#### 14.4. Allocate command buffers
#### 14.5. Create command buffer begin info
#### 14.6. Begin command buffer
#### 14.7. Create render pass begin info
#### 14.8. Record begin render pass command
#### 14.9. Record bind pipeline command
#### 14.10. Record draw command
#### 14.11. Record end render pass command
#### 14.11. End command buffer

## 15. Rendering and presentation
#### 15.1. Create subpass dependencies in render pass creation code
#### 15.2. Create image available and render finsihed semaphores
#### 15.3. Create inflight frame and image fences
#### 15.4. Wait for inflight frame fence
#### 15.5. Acquire swapchain image index
#### 15.6. Wait for inflight image fence
#### 15.7. Create submit info
#### 15.8. Reset inflight frame fence
#### 15.9. Submit command buffer related to aquired image index, on queue
#### 15.10. Present image by index on present queue

## This file may not get updated anymore for future chapters.
