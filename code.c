#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define MAX_SOURCE_SIZE (0x100000)

void loadImageData(const char *filename, uchar4 *image_data, int width, int height) {
    int channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, STBI_rgb_alpha);
    if (data == NULL) {
        fprintf(stderr, "Failed to load image file: %s\n", filename);
        exit(1);
    }
    // Convert to uchar4 array
    for (int i = 0; i < width * height; i++) {
        image_data[i].x = data[i * 4];
        image_data[i].y = data[i * 4 + 1];
        image_data[i].z = data[i * 4 + 2];
        image_data[i].w = data[i * 4 + 3];
    }
    stbi_image_free(data);
}

void saveGrayscaleImage(const char *filename, uchar *grayscale_image, int width, int height) {
    if (!stbi_write_jpg(filename, width, height, 1, grayscale_image, 100)) {
        fprintf(stderr, "Failed to write image file: %s\n", filename);
        exit(1);
    }
}

int main() {
    // Load kernel source code
    FILE *kernel_file;
    char *kernel_source;
    size_t kernel_size;

    kernel_file = fopen("grayscale_kernel.cl", "r");
    if (!kernel_file) {
        fprintf(stderr, "Failed to open kernel file\n");
        return 1;
    }

    kernel_source = (char *)malloc(MAX_SOURCE_SIZE);
    kernel_size = fread(kernel_source, 1, MAX_SOURCE_SIZE, kernel_file);
    fclose(kernel_file);

    // Setup OpenCL environment
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint num_devices;
    cl_uint num_platforms;

    clGetPlatformIDs(1, &platform_id, &num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

    // Load image data and create OpenCL buffers
    // For simplicity, let's assume we have the image data and dimensions already available
    int width, height, channels;
    uchar4 *input_image = (uchar4 *)stbi_load("input_image.jpg", &width, &height, &channels, STBI_rgb_alpha);
    if (input_image == NULL) {
        fprintf(stderr, "Failed to load input image\n");
        return 1;
    }
    int image_size = width * height;

    uchar *output_image = (uchar *)malloc(image_size * sizeof(uchar));

    // Create OpenCL buffers for image data
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size * sizeof(uchar4), input_image, NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_size * sizeof(uchar), NULL, NULL);

    // Create OpenCL program and kernel
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, (const size_t *)&kernel_size, NULL);
    clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "convertToGrayscale", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&width);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&height);

    // Enqueue kernel execution
    size_t global_work_size[2] = {width, height};
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

    // Read back results
    clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0, image_size * sizeof(uchar), output_image, 0, NULL, NULL);

    // Save grayscale images to disk
    saveGrayscaleImage("output_image.jpg", output_image, width, height);

    return 0;
}