/**
 * @example status.c
 *
 * @brief Shows how to use the status enum to check function call failures.
 * */

#include <NanoNet.h>

#include <stdio.h>

int
main()
{
  struct NanoNet_VM* vm = NanoNet_New();

  uint32_t bad_opcode = 0xffffffffU;

  enum NanoNet_Status status = NanoNet_Forward(vm, &bad_opcode, 1);
  if (status != NANONET_OK) {
    printf("Failed to execute forward pass: %s\n", NanoNet_StatusString(status));
  }

  NanoNet_Free(vm);
  return 0;
}
