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

## 4. Physical device and queue families selection
#### 4.1. Enumerate physical devices
#### 4.2. Check for appropriate physical device
##### 4.2.1. Check for appropriate physical device properties
##### 4.2.2. Check for appropriate physical device features
##### 4.2.3. Check for appropriate physical device queue family
#### 4.3. If physical device passes all checks, store physical device and needed queue family indices in relevant structs

## 5. Logical device creation and acquiring queues