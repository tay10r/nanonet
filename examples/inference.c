/**
 * @example inference.c
 *
 * @brief An example that demonstrates how to run a forward pass.
 * */

#include <NanoNet.h>

#include <stdio.h>
#include <stdlib.h>

static void
report_build_error(void* unused, const char* description, const size_t description_len)
{
  printf("%s\n", description);
}

static NANONET_DEF_MODULE(example_module)
{
  uint8_t x = NANONET_INPUT(4, 1);
  x = NANONET_RELU(NANONET_MATMUL(NANONET_PARAM(4, 4), x));
  x = NANONET_RELU(NANONET_MATMUL(NANONET_PARAM(4, 4), x));
  x = NANONET_SIGMOID(NANONET_MATMUL(NANONET_PARAM(4, 4), x));
  return x;
}

static float
rand_float()
{
  const float x = ((float)rand()) / ((float)RAND_MAX);
  return x * 2.0F - 1.0F;
}

int
main()
{
  struct nn_module* m = NULL;

  enum nn_status s = nn_build_module(&m, example_module, /*error_callback_data=*/NULL, report_build_error);
  if (s != NANONET_OK) {
    printf("failed to build module: %s\n", nn_status_string(s));
    return EXIT_FAILURE;
  }

  nn_randomize_params(m, /*seed=*/0);

  struct nn_vm* vm = nn_vm_new();
  if (!vm) {
    printf("failed to allocate VM.\n");
    free(m);
    return EXIT_FAILURE;
  }

  /* Normally there isn't just a single forward pass, there is some sort of loop.
   * We'll pretend that we have to run the network 3 times, but this is an arbitrary number.
   */
  const int num_iterations = 5;

  uint32_t num_opcodes = 0;

  uint8_t num_params = nn_get_num_params(m);

  for (uint8_t i = 0; i < num_params; i++) {
    uint8_t reg = 0;
    uint8_t rows = 0;
    uint8_t cols = 0;
    float* buffer = nn_get_param(m, i, &reg, &rows, &cols);
    nn_set_reg(vm, reg, rows, cols, buffer);
    printf("set parameter[%d] to register %d with shape (%d, %d)\n", (int)i, (int)reg, (int)rows, (int)cols);
  }

  const uint32_t* opcodes = nn_get_module_code(m, &num_opcodes);

  for (int i = 0; i < num_iterations; i++) {

    nn_reset(vm);

    const float input[4] = { rand_float(), rand_float(), rand_float(), rand_float() };

    nn_set_reg(vm, /*reg_index=*/0, 4, 1, input);

    s = nn_forward(vm, opcodes, num_opcodes);
    if (s != NANONET_OK) {
      printf("failed to run forward pass: %s\n", nn_status_string(s));
      break;
    }

    const uint8_t out_reg = nn_get_output_register(m);

    const float* out = nn_get_reg(vm, out_reg);

    printf("result from register %d: %f\n", (int)out_reg, out[0]);
  }

  free(vm);

  free(m);

  return EXIT_SUCCESS;
}
