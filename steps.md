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
#### 6.6. Store required queue handles in an appropriate place