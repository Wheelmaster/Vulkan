#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <fstream>
#include <set>
#include <vector>
#include <array>
#include <algorithm>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>

const int WIDTH = 800;
const int HEIGHT = 600;

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

	size_t fileSize = (size_t) file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

const std::vector<const char*> VALIDATION_LAYERS = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> DEVICE_EXTENSIONS = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct Buffer {
	VkBuffer handle;
	VkDeviceMemory memory;
};

struct Image {
	VkImage handle;
	VkDeviceMemory memory;
};

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

const std::vector<Vertex> VERTICES = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> INDICES = {
    0, 1, 2, 2, 3, 0
};

struct Device {
	VkDevice handle;
	VkQueue queuePresent;
	VkQueue queueGraphics;
};

struct PhysicalDevice {
	VkPhysicalDevice handle;
    int queueFamilyIndexGraphics;
    int queueFamilyIndexPresent;
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
    
bool checkValidationLayerSupport() {
    // Get available layers
	uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

	// Test if all requested layers are present
	for (const char* layerName : VALIDATION_LAYERS) {
		bool layerFound = false;

		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}

		if (!layerFound) {
			return false;
		}
	}

	return true;
}

std::vector<const char*> getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    }

    return extensions;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugReportFlagsEXT flags,
    VkDebugReportObjectTypeEXT objType,
    uint64_t obj,
    size_t location,
    int32_t code,
    const char* layerPrefix,
    const char* msg,
    void* userData) {

    std::cerr << "validation layer: " << msg << std::endl;

    return VK_FALSE;
}

VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback) {
    auto func = (PFN_vkCreateDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugReportCallbackEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}

VkRenderPass createRenderPass(VkFormat swapChainImageFormat, VkDevice device) {
	VkAttachmentDescription colorAttachment = {};
	colorAttachment.format = swapChainImageFormat;
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	
	VkAttachmentReference colorAttachmentRef = {};
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;		

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0;
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 1;
	renderPassInfo.pDependencies = &dependency;

	VkRenderPass renderPass = VK_NULL_HANDLE;
	if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass!");
	}

	return renderPass;
}

VkInstance createInstance() {
	if (enableValidationLayers && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers requested, but not available!");
	}

	// Fill in some information about the application, could be set to anything
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Hello Triangle";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_1;

	// Prepare the instance creation parameters: validation layers + extensions
	VkInstanceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	createInfo.pApplicationInfo = &appInfo;
	
	if (enableValidationLayers) {
		createInfo.enabledLayerCount = (uint32_t)VALIDATION_LAYERS.size();
		createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
	}
	
	const std::vector<const char*> extensions = getRequiredExtensions();
	createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	createInfo.ppEnabledExtensionNames = extensions.data();

	// Create instance
	VkInstance instance = VK_NULL_HANDLE;
	if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("failed to create instance!");
	}

	return instance;
}

// Create a surface tied to the window that Vulkan can use for presenting
VkSurfaceKHR createSurface(VkInstance instance, GLFWwindow *window) {
	VkSurfaceKHR surface = VK_NULL_HANDLE;
	if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface!");
	}

	return surface;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
	uint32_t extensionCount;
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

	std::vector<VkExtensionProperties> availableExtensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

	std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

	for (const auto& extension : availableExtensions) {
		requiredExtensions.erase(extension.extensionName);
	}

	return requiredExtensions.empty();
}

SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
	// Get the supported number of swapchain images and range of resolutions supported by the surface
	SwapChainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	// Get the swapchain formats supported by the surface
	uint32_t formatCount;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
	if (formatCount != 0) {
		details.formats.resize(formatCount);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
	}		

	// Get supported present modes
	uint32_t presentModeCount;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
	if (presentModeCount != 0) {
		details.presentModes.resize(presentModeCount);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
	}

	return details;
}

bool isDeviceSuitable(PhysicalDevice device, VkSurfaceKHR surface) {
	bool extensionsSupported = checkDeviceExtensionSupport(device.handle);
	bool swapChainAdequate = false;
	if (extensionsSupported) {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device.handle, surface);
		swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
	}

	return device.queueFamilyIndexGraphics != -1 && device.queueFamilyIndexPresent != -1 && extensionsSupported && swapChainAdequate;
}

// Pick one of the available physical devices to use
PhysicalDevice pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface) {	
	PhysicalDevice selectedDevice = {};

	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

	if (deviceCount == 0) {
		throw std::runtime_error("failed to find GPUs with Vulkan support");
	}

	std::vector<PhysicalDevice> devices(deviceCount);
	// Init devices array
	{ 
		std::vector<VkPhysicalDevice> handles(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, handles.data());

		for (uint32_t n = 0; n < deviceCount; n++) {
			devices[n].handle = handles[n];
			devices[n].queueFamilyIndexGraphics = -1;
			devices[n].queueFamilyIndexPresent = -1;
		}
	}

	int scoreToBeat = 0;
	for (auto &device : devices) {
		{ // Get queue family indices
			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device.handle, &queueFamilyCount, nullptr);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device.handle, &queueFamilyCount, queueFamilies.data());

			int i = 0;
			for (const auto& queueFamily : queueFamilies) {
				if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					device.queueFamilyIndexGraphics = i;
				}

				VkBool32 presentSupport = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device.handle, i, surface, &presentSupport);

				if (queueFamily.queueCount > 0 && presentSupport) {
					device.queueFamilyIndexPresent = i;
				}

				const bool foundIndices = (device.queueFamilyIndexGraphics != -1 && device.queueFamilyIndexPresent != -1);
				if (foundIndices) {
					break;
				}

				i++;
			}
		}

		if (isDeviceSuitable(device, surface)) {
			// Get suitability score
			int score = 0;
			
			VkPhysicalDeviceProperties properties;
			vkGetPhysicalDeviceProperties(device.handle, &properties);

			if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				score ++;
			}

			// Test if the current physical device is the best one yet
			if (score > scoreToBeat || selectedDevice.handle == VK_NULL_HANDLE) {
				selectedDevice = device;
				scoreToBeat = score;
			}
		}
	}

	if (selectedDevice.handle == VK_NULL_HANDLE) {
		throw std::runtime_error("failed to find a suitable GPU!");
	}

	{ // Output physical device info
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(selectedDevice.handle, &properties);
		std::cout << "Selected physical device: " << properties.deviceName << std::endl;
	}

	return selectedDevice;
}

Device createLogicalDevice(PhysicalDevice physicalDevice, VkSurfaceKHR surface) {
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	{ // Prepare queue creation parameters
		std::set<int> uniqueQueueFamilies = {physicalDevice.queueFamilyIndexGraphics, physicalDevice.queueFamilyIndexPresent}; // Use a set to 
		
		static const float QUEUE_PRIORITY = 1.0f; // Could be anything, only using a single queue per family
		for (int queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &QUEUE_PRIORITY;
			queueCreateInfos.push_back(queueCreateInfo);
		}	
	}
		

	VkDeviceCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	createInfo.pQueueCreateInfos = queueCreateInfos.data();
	createInfo.queueCreateInfoCount = uint32_t(queueCreateInfos.size());
	static const VkPhysicalDeviceFeatures DEVICE_FEATURES = {};
	createInfo.pEnabledFeatures = &DEVICE_FEATURES;
	createInfo.enabledExtensionCount = uint32_t(DEVICE_EXTENSIONS.size());
	createInfo.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

	if (enableValidationLayers) {
		createInfo.enabledLayerCount = uint32_t(VALIDATION_LAYERS.size());
		createInfo.ppEnabledLayerNames = VALIDATION_LAYERS.data();
	} else {
		createInfo.enabledLayerCount = 0;
	}

	Device device = {};
	if (vkCreateDevice(physicalDevice.handle, &createInfo, nullptr, &device.handle) != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device!");
	}

	vkGetDeviceQueue(device.handle, physicalDevice.queueFamilyIndexGraphics, 0, &device.queueGraphics);
	vkGetDeviceQueue(device.handle, physicalDevice.queueFamilyIndexPresent, 0, &device.queuePresent);

	return device;
}

struct Swapchain {
	VkSwapchainKHR handle;
	VkExtent2D extent;
	VkFormat imageFormat;
	std::vector<VkImage> images;
	std::vector<VkImageView> imageViews;	

};

Swapchain createSwapChain(PhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface) {
	// Get details about what swapchain parameters are supported
	SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice.handle, surface);

	// Get surface format
	VkSurfaceFormatKHR surfaceFormat = {};
	{ 
		// Test for the special case when VK_FORMAT_UNDEFINED is the only returned format,
		// then the application can pick any VkFormat
		static const VkFormat FORMAT = VK_FORMAT_B8G8R8A8_UNORM;
		static const VkColorSpaceKHR COLOR_SPACE = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

		const std::vector<VkSurfaceFormatKHR> &formats = swapChainSupport.formats;		
		if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) { 
			surfaceFormat.format = FORMAT;
			surfaceFormat.colorSpace = COLOR_SPACE;
		} else {
			for (const auto& format : formats) {
				if (format.format == FORMAT && format.colorSpace == COLOR_SPACE) {
					surfaceFormat = format;
					break;
				}
			}
		}
	}

	// Get extent
	VkExtent2D extent = {};
	{
		const VkSurfaceCapabilitiesKHR &capabilities = swapChainSupport.capabilities;
		static const int SIZE_UNDEFINED = 0xFFFFFFFF;
		if (capabilities.currentExtent.width == SIZE_UNDEFINED) {
			// Pick a size and make sure it lays within the allowed range
			extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, (uint32_t)WIDTH));
			extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, (uint32_t)HEIGHT));
		} else {
			extent = capabilities.currentExtent; // Use the width and height of the surface
		}
	}

	// Get the number of images
	static const int IMAGE_COUNT_DOUBLE_BUFFERING = 2;
	uint32_t imageCount = IMAGE_COUNT_DOUBLE_BUFFERING;
	if (swapChainSupport.capabilities.maxImageCount != 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
		std::cout << "Requested number of images too high" << std::endl;
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	// Fill in the swapchain creation parameters
	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = surface;
	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	uint32_t queueFamilyIndices[] = {(uint32_t)physicalDevice.queueFamilyIndexGraphics, (uint32_t)physicalDevice.queueFamilyIndexPresent};
	if (physicalDevice.queueFamilyIndexGraphics != physicalDevice.queueFamilyIndexPresent) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = 2;
		createInfo.pQueueFamilyIndices = queueFamilyIndices;
	} else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0; // Optional
		createInfo.pQueueFamilyIndices = nullptr; // Optional
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	createInfo.clipped = VK_TRUE;
	createInfo.oldSwapchain = VK_NULL_HANDLE;

	Swapchain swapchain = {};
	if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain.handle) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain!");
	}

	// Get the swapchain images
	vkGetSwapchainImagesKHR(device, swapchain.handle, &imageCount, nullptr);
	swapchain.images.resize(imageCount);
	vkGetSwapchainImagesKHR(device, swapchain.handle, &imageCount, swapchain.images.data());
	swapchain.imageFormat = surfaceFormat.format;
	swapchain.extent = extent;

	// Create image views for all images
	swapchain.imageViews.resize(imageCount);
	for (size_t i = 0; i < swapchain.images.size(); i++) {
		VkImageViewCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = swapchain.images[i];

		createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		createInfo.format = swapchain.imageFormat;

		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &createInfo, nullptr, &swapchain.imageViews[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image views!");
		}
	}

	return swapchain;
}

std::vector<VkFramebuffer> createFramebuffers(VkDevice device, const Swapchain &swapchain, VkRenderPass renderPass) {
	std::vector<VkFramebuffer> swapChainFramebuffers(swapchain.imageViews.size());

	for (size_t i = 0; i < swapchain.imageViews.size(); i++) {
		VkImageView attachments[] = {
			swapchain.imageViews[i]
		};

		VkFramebufferCreateInfo framebufferInfo = {};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = 1;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = swapchain.extent.width;
		framebufferInfo.height = swapchain.extent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer!");
		}
	}

	return swapChainFramebuffers;
}

VkCommandPool createCommandPool(PhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface) {
	VkCommandPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	poolInfo.queueFamilyIndex = physicalDevice.queueFamilyIndexGraphics;
	poolInfo.flags = 0; // Optional

	VkCommandPool commandPool = VK_NULL_HANDLE;
	if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool!");
	}

	return commandPool;
}

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice) {
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type!");
}

Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkDevice device, VkPhysicalDevice physicalDevice) {
	VkBufferCreateInfo bufferInfo = {};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = usage;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	Buffer buffer = {};
	if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.handle) != VK_SUCCESS) {
		throw std::runtime_error("failed to create buffer!");
	}

	// Allocate buffer memory
	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(device, buffer.handle, &memRequirements);

	VkMemoryAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice);
	if (vkAllocateMemory(device, &allocInfo, nullptr, &buffer.memory) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate buffer memory!");
	}

	vkBindBufferMemory(device, buffer.handle, buffer.memory, 0);

	return buffer;
}

VkCommandBuffer beginSingleTimeCommands(VkCommandPool commandPool, VkDevice device) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkQueue graphicsQueue, VkDevice device, VkCommandPool commandPool) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, VkDevice device, VkCommandPool commandPool, VkQueue graphicsQueue) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device);

	VkBufferImageCopy region = {};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;

	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;

	region.imageOffset = {0, 0, 0};
	region.imageExtent = {
		width,
		height,
		1
	};

	vkCmdCopyBufferToImage(
		commandBuffer,
		buffer,
		image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);

    endSingleTimeCommands(commandBuffer, graphicsQueue, device, commandPool);
}

void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, VkDevice device, VkQueue graphicsQueue, VkCommandPool commandPool) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device);

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	barrier.srcAccessMask = 0; // TODO
	barrier.dstAccessMask = 0; // TODO
	
	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	} else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else {
		throw std::invalid_argument("unsupported layout transition!");
	}

	vkCmdPipelineBarrier(
		commandBuffer,
		sourceStage, destinationStage,
		0,
		0, nullptr,
		0, nullptr,
		1, &barrier
	);

    endSingleTimeCommands(commandBuffer, graphicsQueue, device, commandPool);
}

void copyBuffer(VkCommandPool commandPool, const Device &device, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands(commandPool, device.handle);

    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer, device.queueGraphics, device.handle, commandPool);
}

Image createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkDevice device, VkPhysicalDevice physicalDevice) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	Image image = {};
    if (vkCreateImage(device, &imageInfo, nullptr, &image.handle) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image.handle, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, physicalDevice);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &image.memory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image.handle, image.memory, 0);

	return image;
}

Image createTextureImage(VkDevice device, VkQueue graphicsQueue, VkPhysicalDevice physicalDevice, VkCommandPool commandPool) {
    // Load image
	int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load("textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }

	// Create staging buffer
	Buffer stagingBuffer = createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, device, physicalDevice);

	// Copy image data to staging buffer
	void *data;
	vkMapMemory(device, stagingBuffer.memory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));
	vkUnmapMemory(device, stagingBuffer.memory);

	// Free image
	stbi_image_free(pixels);

	Image textureImage = createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, device, physicalDevice);
	transitionImageLayout(textureImage.handle, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, device, graphicsQueue, commandPool);
	copyBufferToImage(stagingBuffer.handle, textureImage.handle, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), device, commandPool, graphicsQueue);
	transitionImageLayout(textureImage.handle, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, device, graphicsQueue, commandPool);
	
    vkDestroyBuffer(device, stagingBuffer.handle, nullptr);
    vkFreeMemory(device, stagingBuffer.memory, nullptr);

	return textureImage;
}

Buffer createVertexBuffer(Device device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool) {
    VkDeviceSize bufferSize = sizeof(VERTICES[0])*VERTICES.size();

    Buffer stagingBuffer;
    stagingBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, device.handle, physicalDevice);

    void* data;
    vkMapMemory(device.handle, stagingBuffer.memory, 0, bufferSize, 0, &data);
	memcpy(data, VERTICES.data(), (size_t) bufferSize);
    vkUnmapMemory(device.handle, stagingBuffer.memory);

    Buffer vertexBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, device.handle, physicalDevice);

	copyBuffer(commandPool, device, stagingBuffer.handle, vertexBuffer.handle, bufferSize);

    vkDestroyBuffer(device.handle, stagingBuffer.handle, nullptr);
    vkFreeMemory(device.handle, stagingBuffer.memory, nullptr);

	return vertexBuffer;
}

Buffer createIndexBuffer(Device device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool) {
    VkDeviceSize bufferSize = sizeof(INDICES[0])*INDICES.size();

    Buffer stagingBuffer;
    stagingBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, device.handle, physicalDevice);

    void* data;
    vkMapMemory(device.handle, stagingBuffer.memory, 0, bufferSize, 0, &data);
	memcpy(data, INDICES.data(), (size_t) bufferSize);
    vkUnmapMemory(device.handle, stagingBuffer.memory);

    Buffer indexBuffer = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, device.handle, physicalDevice);

	copyBuffer(commandPool, device, stagingBuffer.handle, indexBuffer.handle, bufferSize);

    vkDestroyBuffer(device.handle, stagingBuffer.handle, nullptr);
    vkFreeMemory(device.handle, stagingBuffer.memory, nullptr);

	return indexBuffer;
}

VkDescriptorPool createDescriptorPool(VkDevice device) {
	VkDescriptorPoolSize poolSize = {};
	poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	poolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo = {};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	poolInfo.maxSets = 1;

	VkDescriptorPool descriptorPool;
	if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}

	return descriptorPool;
}

VkDescriptorSet createDescriptorSet(VkDevice device, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorPool descriptorPool, VkBuffer uniformBuffer) {
	VkDescriptorSetLayout layouts[] = {descriptorSetLayout};
	VkDescriptorSetAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = descriptorPool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.pSetLayouts = layouts;

	VkDescriptorSet descriptorSet;
	if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate descriptor set!");
	}

	VkDescriptorBufferInfo bufferInfo = {};
	bufferInfo.buffer = uniformBuffer;
	bufferInfo.offset = 0;
	bufferInfo.range = sizeof(UniformBufferObject);

	VkWriteDescriptorSet descriptorWrite = {};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = descriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.dstArrayElement = 0;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.pBufferInfo = &bufferInfo;
	descriptorWrite.pImageInfo = nullptr; // Optional
	descriptorWrite.pTexelBufferView = nullptr; // Optional

	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

	return descriptorSet;
}

Buffer createUniformBuffer(VkDevice device, VkPhysicalDevice physicalDevice) {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    return createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, device, physicalDevice);
}

std::vector<VkCommandBuffer> createCommandBuffers(VkDevice device, const std::vector<VkFramebuffer> &swapChainFramebuffers, VkCommandPool commandPool, VkRenderPass renderPass, VkExtent2D extent, VkPipeline graphicsPipeline, VkBuffer vertexBuffer, VkBuffer indexBuffer, VkDescriptorSet descriptorSet, VkPipelineLayout pipelineLayout) {
	std::vector<VkCommandBuffer> commandBuffers(swapChainFramebuffers.size());

	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = commandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

	if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers!");
	}

	for (size_t i = 0; i < commandBuffers.size(); i++) {
		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		beginInfo.pInheritanceInfo = nullptr; // Optional

		if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[i];
		renderPassInfo.renderArea.offset = {0, 0};
		renderPassInfo.renderArea.extent = extent;

		VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkBuffer vertexBuffers[] = {vertexBuffer};
		VkDeviceSize offsets[] = {0};
		vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT16);
		vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

		vkCmdDrawIndexed(commandBuffers[i], (uint32_t)INDICES.size(), 1, 0, 0, 0);
		vkCmdEndRenderPass(commandBuffers[i]);

		if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	return commandBuffers;
}

void createSemaphores(VkDevice device, VkSemaphore *imageAvailableSemaphore, VkSemaphore *renderFinishedSemaphore) {
	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, imageAvailableSemaphore) != VK_SUCCESS ||
		vkCreateSemaphore(device, &semaphoreInfo, nullptr, renderFinishedSemaphore) != VK_SUCCESS) {

		throw std::runtime_error("failed to create semaphores!");
	}
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module!");
	}

	return shaderModule;
}

struct GraphicsPipeline {
	VkPipeline handle;
	VkPipelineLayout layout;
};

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
	uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo = {};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &uboLayoutBinding;

	VkDescriptorSetLayout descriptorSetLayout;
	if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}

	return descriptorSetLayout;
}

GraphicsPipeline createGraphicsPipeline(VkDevice device, VkExtent2D extent, VkRenderPass renderPass, VkDescriptorSetLayout descriptorSetLayout) {
	// Create shaders
	const std::vector<char> vertShaderCode = readFile("shaders/vert.spv");
	const std::vector<char> fragShaderCode = readFile("shaders/frag.spv");
	
	const VkShaderModule vertShaderModule = createShaderModule(device, vertShaderCode);
	const VkShaderModule fragShaderModule = createShaderModule(device, fragShaderCode);

	VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
	vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vertShaderStageInfo.module = vertShaderModule;
	vertShaderStageInfo.pName = "main";

	VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
	fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fragShaderStageInfo.module = fragShaderModule;
	fragShaderStageInfo.pName = "main";

	const VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

	// Vertex input
	const VkVertexInputBindingDescription bindingDescription = Vertex::getBindingDescription();
	const std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = Vertex::getAttributeDescriptions();

	VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.vertexAttributeDescriptionCount = (uint32_t)attributeDescriptions.size();
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkViewport viewport = {};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)extent.width;
	viewport.height = (float)extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {};
	scissor.offset = {0, 0};
	scissor.extent = extent;

	VkPipelineViewportStateCreateInfo viewportState = {};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	VkPipelineRasterizationStateCreateInfo rasterizer = {};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthBiasConstantFactor = 0.0f; // Optional
	rasterizer.depthBiasClamp = 0.0f; // Optional
	rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

	VkPipelineMultisampleStateCreateInfo multisampling = {};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	multisampling.minSampleShading = 1.0f; // Optional
	multisampling.pSampleMask = nullptr; // Optional
	multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
	multisampling.alphaToOneEnable = VK_FALSE; // Optional

	VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
	colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
	colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
	colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
	colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

	VkPipelineColorBlendStateCreateInfo colorBlending = {};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f; // Optional
	colorBlending.blendConstants[1] = 0.0f; // Optional
	colorBlending.blendConstants[2] = 0.0f; // Optional
	colorBlending.blendConstants[3] = 0.0f; // Optional

	VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
	pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

	GraphicsPipeline pipeline = {};
	if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipeline.layout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout!");
	}

	VkGraphicsPipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = nullptr; // Optional
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDynamicState = nullptr; // Optional
	pipelineInfo.layout = pipeline.layout;
	pipelineInfo.renderPass = renderPass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
	pipelineInfo.basePipelineIndex = -1; // Optional

	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline.handle) != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline!");
	}

	vkDestroyShaderModule(device, fragShaderModule, nullptr);
	vkDestroyShaderModule(device, vertShaderModule, nullptr);

	return pipeline;
}

GLFWwindow *initWindow() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	return glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
}

class HelloTriangleApplication {
public:
    void run() {
		window = initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    void initVulkan() {
		instance = createInstance();
		setupDebugCallback();
		surface = createSurface(instance, window);
		physicalDevice = pickPhysicalDevice(instance, surface);
		device = createLogicalDevice(physicalDevice, surface);
		swapchain = createSwapChain(physicalDevice, device.handle, surface);		
		renderPass = createRenderPass(swapchain.imageFormat, device.handle);
		descriptorSetLayout = createDescriptorSetLayout(device.handle);
		graphicsPipeline = createGraphicsPipeline(device.handle, swapchain.extent, renderPass, descriptorSetLayout);
		swapChainFramebuffers = createFramebuffers(device.handle, swapchain, renderPass);
		commandPool = createCommandPool(physicalDevice, device.handle, surface);
		textureImage = createTextureImage(device.handle, device.queueGraphics, physicalDevice.handle, commandPool);
		vertexBuffer = createVertexBuffer(device, physicalDevice.handle, commandPool);
		indexBuffer = createIndexBuffer(device, physicalDevice.handle, commandPool);
		uniformBuffer = createUniformBuffer(device.handle, physicalDevice.handle);
		descriptorPool = createDescriptorPool(device.handle);
		descriptorSet = createDescriptorSet(device.handle, descriptorSetLayout, descriptorPool, uniformBuffer.handle);
		commandBuffers = createCommandBuffers(device.handle, swapChainFramebuffers, commandPool, renderPass, swapchain.extent, graphicsPipeline.handle, vertexBuffer.handle, indexBuffer.handle, descriptorSet, graphicsPipeline.layout);
		createSemaphores(device.handle, &imageAvailableSemaphore, &renderFinishedSemaphore);
    }	
	
	void setupDebugCallback() {
		if (!enableValidationLayers) { return; }

		VkDebugReportCallbackCreateInfoEXT createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		createInfo.pfnCallback = debugCallback;

		if (CreateDebugReportCallbackEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug callback!");
		}
	}

    void mainLoop() {
		while(!glfwWindowShouldClose(window) && glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS) {
			glfwPollEvents();

			updateUniformBuffer();
			drawFrame();
		}

	    vkDeviceWaitIdle(device.handle);
    }

	void updateUniformBuffer() {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapchain.extent.width/(float)swapchain.extent.height, 0.1f, 10.0f);

		ubo.proj[1][1] *= -1;

		void *data;
		vkMapMemory(device.handle, uniformBuffer.memory, 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device.handle, uniformBuffer.memory);
	}

	void drawFrame() {
		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device.handle, swapchain.handle, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			std::cout << "Swap chain out date" << std::endl;
			recreateSwapChain();
			return;
		} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		// Do not begin executing the command buffer before this semaphore has been signaled
		const VkSemaphore waitSemaphores[] = {imageAvailableSemaphore}; 
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		
		static const VkPipelineStageFlags WAIT_STAGES[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		submitInfo.pWaitDstStageMask = WAIT_STAGES;
		
		// The command buffer to execute
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

		// Signal this semaphore when all command buffer operations has been completed
		const VkSemaphore signalSemaphores[] = {renderFinishedSemaphore};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores; 
		
		// Submit the command buffer to the graphics queue
		if (vkQueueSubmit(device.queueGraphics, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		// Set present queue parameters
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		const VkSwapchainKHR swapChains[] = {swapchain.handle};
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		// Queue image for presentation
		vkQueuePresentKHR(device.queuePresent, &presentInfo);

		// Wait until all graphics and present commands has been executed
		if (enableValidationLayers) {
			vkQueueWaitIdle(device.queuePresent);
		}
	}

	void cleanupSwapChain() {
		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device.handle, framebuffer, nullptr);
		}
		
		vkFreeCommandBuffers(device.handle, commandPool, (uint32_t)(commandBuffers.size()), commandBuffers.data());

		vkDestroyPipeline(device.handle, graphicsPipeline.handle, nullptr);
		vkDestroyPipelineLayout(device.handle, graphicsPipeline.layout, nullptr);
		vkDestroyRenderPass(device.handle, renderPass, nullptr);
		
		for (auto imageView : swapchain.imageViews) {
			vkDestroyImageView(device.handle, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device.handle, swapchain.handle, nullptr);
	}

	void recreateSwapChain() {
		vkDeviceWaitIdle(device.handle);

		cleanupSwapChain();

		swapchain = createSwapChain(physicalDevice, device.handle, surface);		
		renderPass = createRenderPass(swapchain.imageFormat, device.handle);
		graphicsPipeline = createGraphicsPipeline(device.handle, swapchain.extent, renderPass, descriptorSetLayout);
		swapChainFramebuffers = createFramebuffers(device.handle, swapchain, renderPass);
		commandBuffers = createCommandBuffers(device.handle, swapChainFramebuffers, commandPool, renderPass, swapchain.extent, graphicsPipeline.handle, vertexBuffer.handle, indexBuffer.handle, descriptorSet, graphicsPipeline.layout);
	}

    void cleanup() {
		cleanupSwapChain();

		vkDestroyImage(device.handle, textureImage.handle, nullptr);
		vkFreeMemory(device.handle, textureImage.memory, nullptr);

		vkDestroyDescriptorPool(device.handle, descriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(device.handle, descriptorSetLayout, nullptr);
		vkDestroyBuffer(device.handle, uniformBuffer.handle, nullptr);
		vkFreeMemory(device.handle, uniformBuffer.memory, nullptr);

	    vkDestroyBuffer(device.handle, vertexBuffer.handle, nullptr);
		vkFreeMemory(device.handle, vertexBuffer.memory, nullptr);

	    vkDestroyBuffer(device.handle, indexBuffer.handle, nullptr);
		vkFreeMemory(device.handle, indexBuffer.memory, nullptr);

		vkDestroySemaphore(device.handle, imageAvailableSemaphore, nullptr);
		vkDestroySemaphore(device.handle, renderFinishedSemaphore, nullptr);

		vkDestroyCommandPool(device.handle, commandPool, nullptr);

		vkDestroyDevice(device.handle, nullptr);

		if (enableValidationLayers) {
			DestroyDebugReportCallbackEXT(instance, callback, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		
		glfwDestroyWindow(window);

		glfwTerminate();
    }

	GLFWwindow *window;
	VkInstance instance;
	VkDebugReportCallbackEXT callback;
	VkSurfaceKHR surface;
	PhysicalDevice physicalDevice;
	Device device;
	Swapchain swapchain;		
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	GraphicsPipeline graphicsPipeline;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;
	Buffer vertexBuffer;
	Buffer indexBuffer;
	Buffer uniformBuffer;
	VkDescriptorPool descriptorPool;
	VkDescriptorSet descriptorSet;
	Image textureImage;
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}