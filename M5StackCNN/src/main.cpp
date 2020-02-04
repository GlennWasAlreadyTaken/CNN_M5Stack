#include <Arduino.h>
#include <math.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "good_different6_model_data.h"
#include <M5Stack.h>

#define BLACK 0x0000
#define WHITE 0xFFFF
#define TFT_GREY 0x5AEB

// We need to preallocate a certain amount of memory for input, output, and intermediate arrays. 
// This is provided as a uint8_t array of size tensor_arena_size:
constexpr int tensor_pool_size = 100 * 1024;
//uint8_t tensor_pool[tensor_pool_size];

const tflite::Model *model;

tflite::MicroInterpreter *interpreter;

TfLiteTensor *input;
TfLiteTensor *output;

int16_t user_input[128 * 9] = {-282, 990, 227, 872, 324, 171, 456, 885, 194,
                                        -282, 980, 207, 845, 324, 171, 446, 876, 184,
                                        -303, 980, 227, 845, 324, 171, 446, 876, 184,
                                        -282, 970, 217, 854, 333, 161, 456, 885, 155,
                                        -292, 990, 227, 872, 314, 161, 456, 876, 174,
                                        -292, 990, 237, 845, 314, 181, 446, 885, 184,
                                        -282, 970, 217, 863, 342, 141, 456, 885, 184,
                                        -282, 980, 207, 854, 324, 161, 466, 876, 194,
                                        -292, 970, 217, 854, 324, 161, 475, 876, 184,
                                        -292, 980, 227, 872, 324, 161, 456, 885, 174,
                                        -282, 980, 207, 872, 333, 151, 466, 876, 184,
                                        -282, 990, 198, 854, 324, 171, 456, 885, 194,
                                        -292, 970, 217, 863, 324, 161, 475, 885, 165,
                                        -272, 980, 207, 863, 324, 161, 466, 866, 184,
                                        -303, 970, 217, 872, 342, 141, 456, 885, 194,
                                        -282, 980, 207, 863, 305, 171, 446, 885, 184,
                                        -282, 970, 227, 872, 324, 151, 466, 866, 203,
                                        -292, 970, 237, 872, 324, 151, 456, 876, 203,
                                        -292, 990, 217, 863, 324, 171, 446, 885, 174,
                                        -292, 980, 217, 872, 314, 171, 466, 866, 194,
                                        -292, 970, 207, 872, 333, 161, 446, 876, 174,
                                        -282, 970, 207, 872, 314, 181, 466, 895, 174,
                                        -292, 980, 207, 854, 324, 171, 456, 876, 184,
                                        -282, 1000, 207, 872, 324, 151, 475, 847, 194,
                                        -292, 970, 217, 872, 324, 191, 466, 885, 165,
                                        -292, 980, 237, 863, 324, 181, 466, 866, 184,
                                        -282, 970, 237, 863, 324, 181, 466, 885, 194,
                                        -282, 980, 227, 863, 314, 171, 466, 895, 213,
                                        -282, 980, 198, 863, 324, 151, 466, 866, 174,
                                        -292, 970, 207, 863, 324, 171, 456, 895, 203,
                                        -282, 970, 217, 872, 324, 151, 446, 885, 165,
                                        -282, 980, 237, 872, 305, 171, 466, 876, 174,
                                        -303, 980, 237, 863, 333, 161, 466, 876, 174,
                                        -282, 970, 217, 872, 314, 161, 446, 885, 155,
                                        -282, 970, 217, 881, 324, 181, 466, 876, 194,
                                        -292, 970, 207, 863, 333, 171, 456, 885, 194,
                                        -282, 970, 207, 881, 314, 181, 466, 885, 184,
                                        -303, 980, 207, 872, 324, 161, 466, 866, 184,
                                        -282, 980, 227, 872, 314, 171, 456, 885, 194,
                                        -282, 960, 217, 854, 333, 171, 456, 876, 165,
                                        -303, 980, 237, 845, 324, 161, 466, 876, 184,
                                        -282, 980, 217, 872, 324, 161, 456, 885, 194,
                                        -282, 960, 227, 854, 333, 161, 466, 895, 155,
                                        -282, 980, 207, 872, 324, 151, 475, 876, 184,
                                        -282, 970, 227, 854, 314, 181, 466, 876, 184,
                                        -292, 970, 217, 872, 305, 181, 466, 885, 174,
                                        -303, 980, 188, 872, 324, 181, 456, 866, 184,
                                        -292, 990, 198, 872, 324, 151, 466, 876, 203,
                                        -282, 990, 198, 872, 314, 171, 436, 895, 184,
                                        -292, 980, 217, 854, 324, 181, 466, 876, 184,
                                        -282, 980, 227, 872, 314, 151, 456, 876, 203,
                                        -282, 980, 198, 872, 305, 171, 466, 885, 174,
                                        -282, 960, 227, 854, 324, 171, 466, 876, 184,
                                        -303, 990, 227, 872, 324, 181, 456, 885, 184,
                                        -272, 970, 227, 872, 324, 181, 466, 885, 174,
                                        -282, 980, 227, 854, 342, 151, 466, 866, 184,
                                        -292, 990, 217, 863, 324, 171, 466, 885, 194,
                                        -282, 960, 217, 863, 333, 151, 466, 885, 174,
                                        -282, 970, 207, 872, 324, 171, 466, 866, 174,
                                        -282, 980, 207, 881, 305, 181, 456, 876, 184,
                                        -272, 970, 217, 881, 305, 181, 446, 885, 165,
                                        -282, 990, 207, 863, 333, 141, 466, 885, 184,
                                        -282, 970, 227, 854, 314, 161, 456, 885, 194,
                                        -282, 980, 227, 881, 305, 171, 466, 885, 165,
                                        -292, 990, 188, 872, 314, 171, 466, 866, 165,
                                        -292, 980, 217, 872, 324, 161, 466, 876, 184,
                                        -282, 980, 207, 872, 324, 161, 466, 885, 174,
                                        -272, 980, 217, 863, 324, 171, 466, 866, 174,
                                        -292, 980, 217, 872, 333, 151, 456, 885, 194,
                                        -292, 980, 207, 863, 305, 171, 466, 895, 174,
                                        -292, 980, 207, 863, 333, 191, 466, 895, 174,
                                        -292, 970, 217, 872, 333, 191, 456, 885, 174,
                                        -282, 980, 207, 872, 324, 191, 456, 885, 174,
                                        -292, 990, 217, 854, 324, 161, 446, 895, 174,
                                        -292, 990, 217, 881, 314, 151, 466, 866, 194,
                                        -272, 990, 178, 872, 324, 161, 456, 885, 203,
                                        -282, 960, 227, 863, 314, 171, 466, 885, 165,
                                        -282, 980, 227, 854, 324, 171, 466, 885, 194,
                                        -303, 990, 217, 872, 333, 151, 466, 904, 165,
                                        -292, 980, 188, 881, 314, 181, 466, 904, 165,
                                        -292, 980, 188, 854, 333, 171, 475, 876, 184,
                                        -292, 990, 207, 863, 324, 171, 466, 876, 184,
                                        -292, 990, 207, 863, 314, 171, 466, 866, 194,
                                        -292, 970, 217, 854, 314, 151, 466, 866, 194,
                                        -282, 980, 217, 881, 314, 181, 456, 876, 203,
                                        -292, 980, 217, 872, 324, 191, 466, 895, 165,
                                        -272, 970, 217, 872, 324, 161, 446, 885, 174,
                                        -282, 990, 227, 854, 324, 161, 446, 885, 174,
                                        -282, 980, 188, 872, 333, 151, 456, 885, 145,
                                        -303, 970, 217, 881, 314, 171, 475, 866, 184,
                                        -282, 970, 207, 863, 333, 171, 456, 885, 203,
                                        -282, 970, 207, 863, 333, 161, 466, 885, 174,
                                        -292, 980, 227, 863, 314, 181, 466, 876, 194,
                                        -292, 980, 207, 863, 333, 171, 446, 885, 184,
                                        -292, 980, 227, 863, 324, 191, 466, 895, 165,
                                        -292, 970, 217, 863, 324, 171, 466, 876, 165,
                                        -282, 970, 227, 881, 314, 171, 456, 885, 194,
                                        -292, 980, 217, 854, 324, 181, 466, 895, 155,
                                        -292, 980, 207, 863, 324, 151, 466, 876, 213,
                                        -282, 980, 227, 872, 314, 181, 456, 876, 174,
                                        -292, 970, 217, 872, 305, 181, 446, 895, 165,
                                        -292, 970, 237, 872, 333, 181, 475, 876, 203,
                                        -282, 960, 217, 872, 324, 161, 456, 885, 174,
                                        -282, 990, 207, 863, 333, 171, 446, 876, 184,
                                        -292, 980, 217, 845, 333, 191, 456, 885, 194,
                                        -292, 980, 217, 872, 333, 181, 446, 876, 184,
                                        -282, 970, 217, 872, 314, 171, 466, 885, 174,
                                        -292, 990, 227, 863, 333, 191, 466, 866, 184,
                                        -282, 990, 207, 854, 333, 171, 456, 895, 184,
                                        -282, 970, 217, 872, 314, 171, 466, 895, 174,
                                        -282, 960, 227, 854, 324, 191, 475, 866, 184,
                                        -282, 980, 198, 881, 314, 161, 456, 876, 184,
                                        -303, 980, 227, 872, 314, 161, 466, 885, 165,
                                        -292, 990, 217, 863, 324, 171, 475, 866, 184,
                                        -272, 990, 198, 872, 324, 171, 446, 885, 174,
                                        -282, 990, 217, 872, 324, 171, 456, 895, 174,
                                        -282, 970, 198, 863, 333, 141, 466, 876, 184,
                                        -303, 980, 237, 863, 314, 171, 456, 885, 174,
                                        -282, 980, 207, 854, 333, 181, 466, 895, 165,
                                        -282, 970, 217, 863, 314, 151, 466, 866, 184,
                                        -282, 980, 217, 872, 305, 171, 446, 885, 184,
                                        -292, 990, 198, 863, 314, 181, 446, 895, 174,
                                        -292, 970, 217, 854, 333, 151, 456, 876, 174,
                                        -303, 980, 198, 872, 324, 181, 446, 885, 203,
                                        -282, 970, 217, 854, 333, 171, 456, 885, 184,
                                        -292, 980, 227, 881, 314, 161, 466, 876, 194,
                                        -282, 980, 217, 863, 324, 161, 485, 857, 184,
                                        -282, 970, 227, 872, 324, 171, 475, 895, 165};


void setup() {
  M5.begin();
  Serial.begin(9600);

  // Display settings:
  // Fill screen with grey so we can see the effect of printing with and without
  // a background colour defined
  M5.Lcd.fillScreen(TFT_GREY);

  // Set "cursor" at top left corner of display (0,0) and select font 2
  // (cursor will move to next line automatically during printing with 'M5.Lcd.println'
  //  or stay on the line is there is room for the text with M5.Lcd.print)
  M5.Lcd.setCursor(0, 0, 2);
  // Set the font colour to be white with a black background, set text size multiplier to 1
  M5.Lcd.setTextColor(TFT_WHITE, TFT_BLACK);
  M5.Lcd.setTextSize(1);

  // Dynamic allocation of the tensor pool
  uint8_t* tensor_pool = new uint8_t[tensor_pool_size];

  M5.Lcd.println("Loading Tensorflow model...");
  model = tflite::GetModel(good_different6_model_tflite);
  M5.Lcd.println("Model loaded!");

  // Used by the interpreter to access the operations that are used by the model:
  static tflite::ops::micro::AllOpsResolver resolver;

  static tflite::ErrorReporter *error_reporter;
  static tflite::MicroErrorReporter micro_error;
  error_reporter = &micro_error;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_pool, tensor_pool_size, error_reporter);

  interpreter = &static_interpreter;

  Serial.println("Allocating tensors to memory pool");
  TfLiteStatus err = interpreter->AllocateTensors();
  if (err != kTfLiteOk)
  {
    M5.Lcd.println("There was an error allocating the memory...ooof");
    M5.Lcd.println(err);
    return;
  }

  // The MicroInterpreter instance can provide us with a pointer to the model's input tensor by calling .input(0), *
  // where 0 represents the first (and only) input tensor: input = interpreter->input(0);
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Make sure the input has the properties we expect
  //TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  // The property "dims" tells us the tensor's shape. It has one element for
  // each dimension. Our input is a 2D tensor containing 128 elements of 9 values, so "dims"
  // should have size 2.
  //TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).
  //TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  //TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
  // The input is a 32 bit floating point value
  //TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

  M5.Lcd.println("Starting inferences...");
  M5.Lcd.println("Test of FOG Detection");
}

void loop()
{
  //M5.update();

  //M5.Lcd.fillCircle(int(user_input * 50 + 3), int(140 - output->data.f[0] * 100), 8, BLACK);
  /*
  user_input = user_input + 0.1;
  if (user_input > 6.28)
  {
    user_input = 0;
  }
  
  */

  input->data.i16 = user_input;

  if (interpreter->Invoke() != kTfLiteOk)
  {
    M5.Lcd.println("There was an error invoking the interpreter!");
    return;
  }

  // Obtain the output value from the tensor
  float value = output->data.f[0];

  M5.Lcd.print("Input: ");
  M5.Lcd.println(value);
  // Check that the output value is within 0.05 of the expected value
  //TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);

  //M5.Lcd.fillCircle(int(user_input * 50 + 3), int(140 - output->data.f[0] * 100), 8, WHITE);
  delay(1000);
}