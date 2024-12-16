/**
 * @example custom_allocator.c
 *
 * @brief this is an example on how to use custom memory allocation with the library.
 * */

#include <NanoNet.h>

#include <string.h>

/**
 * @brief This is just a simple memory allocator, just meant to prove that you can use one with the library.
 * */
struct CustomAllocator
{
  /**
   * @brief Whether or not the buffer is being used. This is an extreme over-simplification of memory allocators, but
   *        it is just meant for the purpose of this example.
   * */
  int used;

  /**
   * @brief A buffer with one mebibyte of space, enough to allocate the virtual machine.
   * */
  char buffer[1024 * 1024];
};

/**
 * @brief The custom memory allocator singleton.
 * */
static struct CustomAllocator custom_allocator = { 0 };

static void*
custom_malloc(size_t s)
{
  if (custom_allocator.used) {
    return NULL;
  }
  if (s > sizeof(custom_allocator.buffer)) {
    return NULL;
  }
  custom_allocator.used = 1;
  return &custom_allocator.buffer[0];
}

static void
custom_free(void* ptr)
{
  if (ptr == ((void*)custom_allocator.buffer)) {
    custom_allocator.used = 0;
  }
}

int
main()
{
  size_t s = NanoNet_VMSize();
  struct NanoNet_VM* vm = custom_malloc(s);
  memset(vm, 0, NanoNet_VMSize());
  /* At this point, you can use the VM as you normally would. No need to call @ref NanoNet_New or @ref NanoNet_Free */
  custom_free(vm);
  return 0;
}
