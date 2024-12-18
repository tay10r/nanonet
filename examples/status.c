/**
 * @example status.c
 *
 * @brief Shows how to use the status enum to check function call failures.
 * */

#include <NanoNet.h>

#include <stdio.h>
#include <stdlib.h>

int
main()
{
  struct nn_vm* vm = nn_vm_new();

  uint32_t bad_opcode = 0xffffffffU;

  enum nn_status status = nn_forward(vm, &bad_opcode, 1);
  if (status != NANONET_OK) {
    printf("Failed to execute forward pass: %s\n", nn_status_string(status));
  }

  free(vm);

  return 0;
}
