_kernel void convertToGrayscale(_global uchar4 *inputImage, __global uchar *outputImage, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int index = y * width + x;
    uchar4 pixel = inputImage[index];

    // Convert RGB to grayscale using luminance formula
    float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    outputImage[index] = (uchar)gray;
}