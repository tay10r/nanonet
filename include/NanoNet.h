/**
 * @file NanoNet.h
 *
 * @brief The header file containing the API declarations and (optionally) the implementation of the library.
 * */

#pragma once

#include <stddef.h>
#include <stdint.h>

/**
 * @brief This is the major version number of the library.
 *
 * @details If this version number changes, that means there was a change made to the library that is not backwards
 *          compatible on a compilation level. For example, the major version number may change when an existing
 *          function changes the type of one of its arguments.
 * */
#define NANONET_VERSION_MAJOR 1

/**
 * @brief this is the minor version number of the library.
 *
 * @details If this version number changes, that means a feature was added to the library. A change in this version
 *          number does not break backwards compatibility.
 * */
#define NANONET_VERSION_MINOR 0

/**
 * @brief This is the combined major and minor version numbers of the library. The major version has the most
 *        significant bits.
 * */
#define NANONET_VERSION ((NANONET_VERSION_MAJOR << 8) | NANONET_VERSION)

#ifdef NANONET_STATIC
#define NANONET_FUNC static
#endif

#ifndef NANONET_FUNC
/**
 * @brief This macro can be defined in order to add attributes to each of the public functions in the library.
 *        By default, it does nothing.
 * */
#define NANONET_FUNC
#endif

#ifndef NANONET_INFERENCE_ONLY
/**
 * @brief This can be defined as 1 in order to disable all functions and
 *        variables used for training the network. This can reduce the amount
 *        of memory needed to use the library, since a buffer would not be needed
 *        for computing gradients.
 * */
#define NANONET_INFERENCE_ONLY 0
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @brief This enumeration is used to indicate the success or failure of a function call in this library.
   * */
  enum NanoNet_Status
  {
    /**
     * @brief Indicates that the function call was successful.
     * */
    NANONET_OK,
    /**
     * @brief Indicates that a function call failed because there was not enough memory to complete it.
     * */
    NANONET_NO_MEMORY,
    /**
     * @brief Indicates that a function call failed due to an incorrect number of rows or columns in a matrix.
     * */
    NANONET_SHAPE_ERROR,
    /**
     * @brief Indicates that a function call failed due to an malformed opcode.
     * */
    NANONET_BAD_OPCODE
  };

  /**
   * @brief Converts a status code into a human-readable error message.
   *
   * @param s The status code to convert to a message.
   *
   * @return A null-terminated string that describes the status code.
   * */
  NANONET_FUNC const char* NanoNet_StatusString(enum NanoNet_Status s);

  /**
   * @brief This is the opaque structure that represents the virtual machine.
   *
   * @details The virtual machine takes a series of opcodes and executes them in order
   *          to complete the forward and backward passes of the network. Internally
   *          it has one or two floating point buffers and a set of registers used for
   *          executing the opcodes.
   * */
  struct NanoNet_VM;

  /**
   * @brief Indicates the number of bytes in the VM.
   *
   * @details This can be used if you would like to avoid using malloc and free, which are the functions used by the
   *          library in order to allocate the memory for a VM. If you do not want to use those functions, you can use
   *          this function to find out how many bytes your custom memory allocation method needs to store the VM in
   *          memory. Once you have got a pointer to a buffer with this many bytes, you can just memset it to zero in
   *          order to initialize the VM.
   * */
  NANONET_FUNC size_t NanoNet_VMSize(void);

  /**
   * @brief Allocates a new virtual machine for executing the IR modules for this library.
   *
   * @return A pointer to the VM, if the memory allocation was successful. Otherwise, a null pointer is returned.
   * */
  NANONET_FUNC struct NanoNet_VM* NanoNet_New(void);

  /**
   * @brief Releases the memory allocated by the VM.
   *
   * @param self A pointer to the VM to release the memory of.
   * */
  NANONET_FUNC void NanoNet_Free(struct NanoNet_VM* self);

  /**
   * @brief Executes the forward pass of an IR module.
   *
   * @param self A pointer to the VM to execute the module with.
   *
   * @param opcodes A pointer to the array of opcodes associated with the module.
   *
   * @param num_opcodes The number of opcodes in the module.
   *
   * @return A status code to indicate whether this forward pass was successful.
   *
   * @note Be sure to call @ref NanoNet_Reset when you want to call this function again, unless you are continuing the
   *       forward pass with additional opcodes (such as the opcodes for computing loss).
   * */
  NANONET_FUNC enum NanoNet_Status NanoNet_Forward(struct NanoNet_VM* self,
                                                   const uint32_t* opcodes,
                                                   uint32_t num_opcodes);

#if NANONET_INFERENCE_ONLY == 0
  /**
   * @brief Runs the backwards pass of an IR module.
   *
   * @details What this function does is execute the opcodes in reverse, computing the gradients along the way.
   *          This means that before calling this function, the forward pass must have first been called. The gradients
   *          can then be used to update the weights by calling @ref NanoNet_GradientDescent.
   *
   * @param self The VM that executed the forward pass.
   *
   * @param opcodes The opcodes of the forward pass. These get executed in reverse.
   *
   * @param num_opcodes The number of opcodes in the IR module.
   *
   * @return A status code to indicate whether or not the backwards pass was successful.
   * */
  NANONET_FUNC enum NanoNet_Status NanoNet_Backward(struct NanoNet_VM* self,
                                                    const uint32_t* opcodes,
                                                    uint32_t num_opcodes);

  /**
   * @brief Subtracts a fraction of the gradient from the matrix at the given register and places the result
   *        in @p weights.
   *
   * @param self The VM containing the gradient register.
   *
   * @param reg The index of the register to apply gradient descent on.
   *
   * @param learning_rate The fraction of the gradient to subtract from the original matrix.
   *
   * @param weights A pointer to place the result data into. This must be equal to the size of the matrix at the given
   *                register.
   * */
  NANONET_FUNC void NanoNet_GradientDescent(struct NanoNet_VM* self, uint8_t reg, float learning_rate, float* weights);
#endif

  /**
   * @brief Resets the internal memory allocator, making room for a new call to @ref NanoNet_Forward.
   *
   * @details When executing a forward and backward pass, an internal memory buffer is used to make room for the
   *          intermediate results and the outputs. If you want to call the forward and backward passes again, you will
   *          need to restore the space to that buffer so that it can be used again. This function clears that buffer so
   *          that it can be used again.
   * */
  NANONET_FUNC void NanoNet_Reset(struct NanoNet_VM* self);

  /**
   * @brief Sets the data at a given register in the VM.
   *
   * @details Before running a forward pass, you will probably need to set the input matrices for the network. To do
   *          that, you need to call this function and set the shape and coefficients of the matrix.
   *
   * @param self The VM being used to execute an IR module.
   *
   * @param reg_index The index of the register to set the data of.
   *
   * @param rows The number of rows to set this register to.
   *
   * @param cols The number of columns to set this register to.
   *
   * @param data A pointer to the matrix coefficients.
   *
   * @return A status code to indicate whether or not the register was set successfully.
   * */
  NANONET_FUNC enum NanoNet_Status NanoNet_SetRegData(struct NanoNet_VM* self,
                                                      uint8_t reg_index,
                                                      uint8_t rows,
                                                      uint8_t cols,
                                                      const float* data);

  /**
   * @brief Gets a pointer to the data at a given register.
   *
   * @details After running a forward pass with an IR module, you may want to grab the results in order to do something
   *          with them. Calling this function will get you a pointer to any one of the computed values so that you can
   *          do something with it.
   *
   * @param self The VM that executed the forward pass.
   *
   * @param reg_index The index of the register to get the data from.
   * */
  NANONET_FUNC const float* NanoNet_GetRegData(const struct NanoNet_VM* self, uint8_t reg_index);

#ifdef __cplusplus
} /* extern "C" */
#endif

#ifdef NANONET_IMPLEMENTATION

#ifndef NANONET_BUFFER_SIZE
#define NANONET_BUFFER_SIZE 65536
#endif

#ifndef NANONET_REGS
#define NANONET_REGS 256
#endif

#define NANONET_UNUSED(var) ((void)var)

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum NanoNet_Op
  {
    NANONET_OP_MATMUL = 0x01,
    NANONET_OP_ADD = 0x02,
    NANONET_OP_MUL = 0x03,
    NANONET_OP_ROWCAT = 0x04,
    NANONET_OP_COLCAT = 0x05,
    NANONET_OP_MSE = 0x21,
    NANONET_OP_SIGMOID = 0x41,
    NANONET_OP_RELU = 0x42,
    NANONET_OP_TANH = 0x43,
  };

  struct NanoNet_Reg
  {
    uint8_t rows;

    uint8_t cols;

    uint16_t offset;
  };

  static inline int NanoNet_SameShape(const struct NanoNet_Reg* a, const struct NanoNet_Reg* b)
  {
    return (a->rows == b->rows) && (a->cols == b->cols);
  }

  struct NanoNet_VM
  {
    float buffer[NANONET_BUFFER_SIZE];

    struct NanoNet_Reg regs[NANONET_REGS];

    uint16_t buffer_offset;

#if NANONET_INFERENCE_ONLY == 0
    float grad_buffer[NANONET_BUFFER_SIZE];

    uint16_t grad_offset;
#endif
  };

  NANONET_FUNC size_t NanoNet_VMSize(void)
  {
    return sizeof(struct NanoNet_VM);
  }

  NANONET_FUNC const char* NanoNet_StatusString(enum NanoNet_Status s)
  {
    switch (s) {
      case NANONET_OK:
        return "Ok";
      case NANONET_NO_MEMORY:
        return "No Memory";
      case NANONET_SHAPE_ERROR:
        return "Shape Error";
      case NANONET_BAD_OPCODE:
        return "Bad Opcode";
    }
    return "Unknown Status Code";
  }

  NANONET_FUNC struct NanoNet_VM* NanoNet_New(void)
  {
    return (struct NanoNet_VM*)calloc(1, NanoNet_VMSize());
  }

  NANONET_FUNC void NanoNet_Free(struct NanoNet_VM* self)
  {
    free(self);
  }

  static inline uint16_t NanoNet_MemoryRemaining(const struct NanoNet_VM* self)
  {
    return 0xffffU - self->buffer_offset;
  }

  static inline float* NanoNet_AllocReg(struct NanoNet_VM* self,
                                        struct NanoNet_Reg* reg,
                                        const uint8_t rows,
                                        const uint8_t cols)
  {
    const uint16_t size = ((uint16_t)rows) * ((uint16_t)cols);
    if (NanoNet_MemoryRemaining(self) < size) {
      return NULL;
    }
    reg->rows = rows;
    reg->cols = cols;
    reg->offset = self->buffer_offset;
    float* ptr = &self->buffer[self->buffer_offset];
    self->buffer_offset += size;
    return ptr;
  }

  static inline enum NanoNet_Status NanoNet_MatMul(struct NanoNet_VM* self,
                                                   const struct NanoNet_Reg* op1,
                                                   const struct NanoNet_Reg* op2,
                                                   struct NanoNet_Reg* dst)
  {
    float* C = NanoNet_AllocReg(self, dst, op1->rows, op2->cols);
    if (!C) {
      return NANONET_NO_MEMORY;
    }

    memset(C, 0, ((size_t)dst->rows) * ((size_t)dst->cols) * sizeof(float));

    const float* A = &self->buffer[op1->offset];
    const float* B = &self->buffer[op2->offset];

    for (uint32_t j = 0; j < op1->rows; j++) {
      for (uint32_t k = 0; k < op2->cols; k++) {
        const float tmp = B[k + j * op2->cols];
        for (uint32_t i = 0; i < op1->rows; i++) {
          C[i + j * op1->rows] += A[k * op1->cols + i] * tmp;
        }
      }
    }

    return NANONET_OK;
  }

  static inline enum NanoNet_Status NanoNet_Add(struct NanoNet_VM* self,
                                                const struct NanoNet_Reg* op1,
                                                const struct NanoNet_Reg* op2,
                                                struct NanoNet_Reg* dst)
  {
    if (!NanoNet_SameShape(op1, op2)) {
      return NANONET_SHAPE_ERROR;
    }

    float* result = NanoNet_AllocReg(self, dst, op1->rows, op1->cols);
    if (!result) {
      return NANONET_NO_MEMORY;
    }

    const uint16_t size = ((uint16_t)op1->rows) * ((uint16_t)op1->cols);

    const float* a = &self->buffer[op1->offset];
    const float* b = &self->buffer[op2->offset];

    for (uint16_t i = 0; i < size; i++) {
      result[i] = a[i] + b[i];
    }

    return NANONET_OK;
  }

  static inline enum NanoNet_Status NanoNet_Mul(struct NanoNet_VM* self,
                                                const struct NanoNet_Reg* op1,
                                                const struct NanoNet_Reg* op2,
                                                struct NanoNet_Reg* dst)
  {
    if (!NanoNet_SameShape(op1, op2)) {
      return NANONET_SHAPE_ERROR;
    }

    float* result = NanoNet_AllocReg(self, dst, op1->rows, op1->cols);
    if (!result) {
      return NANONET_NO_MEMORY;
    }

    const uint16_t size = ((uint16_t)op1->rows) * ((uint16_t)op1->cols);

    const float* a = &self->buffer[op1->offset];
    const float* b = &self->buffer[op2->offset];

    for (uint16_t i = 0; i < size; i++) {
      result[i] = a[i] * b[i];
    }

    return NANONET_OK;
  }

  static inline enum NanoNet_Status NanoNet_MSE(struct NanoNet_VM* self,
                                                const struct NanoNet_Reg* op1,
                                                const struct NanoNet_Reg* op2,
                                                struct NanoNet_Reg* dst)
  {
    if (!NanoNet_SameShape(op1, op2)) {
      return NANONET_SHAPE_ERROR;
    }

    float* result = NanoNet_AllocReg(self, dst, 1, 1);
    if (!result) {
      return NANONET_NO_MEMORY;
    }

    const uint16_t size = ((uint16_t)op1->rows) * ((uint16_t)op1->cols);

    const float* a = &self->buffer[op1->offset];
    const float* b = &self->buffer[op2->offset];

    float sum = 0.0F;
    for (uint16_t i = 0; i < size; i++) {
      const float delta = a[i] - b[i];
      sum += delta * delta;
    }
    result[0] = sum / ((float)size);

    return NANONET_OK;
  }

  static inline enum NanoNet_Status NanoNet_ReLU(struct NanoNet_VM* self,
                                                 const struct NanoNet_Reg* op1,
                                                 struct NanoNet_Reg* dst)
  {
    float* result = NanoNet_AllocReg(self, dst, op1->rows, op1->cols);
    if (!result) {
      return NANONET_NO_MEMORY;
    }

    const uint16_t size = ((uint16_t)op1->rows) * ((uint16_t)op1->cols);

    const float* in = &self->buffer[op1->offset];

    for (uint16_t i = 0; i < size; i++) {
      const float x = in[i];
      result[i] = (x > 0.0F) ? x : 0.0F;
    }

    return NANONET_OK;
  }

  NANONET_FUNC enum NanoNet_Status NanoNet_Forward(struct NanoNet_VM* self,
                                                   const uint32_t* opcodes,
                                                   uint32_t num_opcodes)
  {
    enum NanoNet_Status status = NANONET_OK;

    for (uint32_t i = 0; i < num_opcodes; i++) {

      const struct NanoNet_Reg* op1 = &self->regs[(opcodes[i] >> 16) & 0xff];
      const struct NanoNet_Reg* op2 = &self->regs[(opcodes[i] >> 8) & 0xff];

      struct NanoNet_Reg* dst = &self->regs[opcodes[i] & 0xff];

      const uint8_t op = (opcodes[i] >> 24) & 0xff;

      switch ((enum NanoNet_Op)op) {
        case NANONET_OP_MATMUL:
          status = NanoNet_MatMul(self, op1, op2, dst);
          break;
        case NANONET_OP_ADD:
          status = NanoNet_Add(self, op1, op2, dst);
          break;
        case NANONET_OP_MUL:
          status = NanoNet_Mul(self, op1, op2, dst);
          break;
        case NANONET_OP_ROWCAT:
          break;
        case NANONET_OP_COLCAT:
          break;
        case NANONET_OP_MSE:
          status = NanoNet_MSE(self, op1, op2, dst);
          break;
        case NANONET_OP_SIGMOID:
          break;
        case NANONET_OP_RELU:
          status = NanoNet_ReLU(self, op1, dst);
          break;
        case NANONET_OP_TANH:
          break;
        default:
          status = NANONET_BAD_OPCODE;
          break;
      }

      if (status != NANONET_OK) {
        break;
      }
    }

    return status;
  }

#if NANONET_INFERENCE_ONLY == 0

  NANONET_FUNC enum NanoNet_Status NanoNet_Back_MatMul(struct NanoNet_VM* self,
                                                       const struct NanoNet_Reg* a,
                                                       const struct NanoNet_Reg* b,
                                                       const struct NanoNet_Reg* c)
  {
    const uint32_t m = (uint32_t)a->rows;
    const uint32_t n = (uint32_t)a->cols;
    const uint32_t p = (uint32_t)b->cols;

    /**
     * Assume the forward pass is: C = A x B
     *
     * We're trying to solve for:
     *
     * dL / dA -> The change in loss with respect to A
     * dL / dB -> The change in loss with respect to B
     *
     * And we know that:
     *   dL / dA =       dL / dC x B^T
     *   dL / dB = A^T x dL / dC
     *
     * Where:
     *   A^T is the transpose of T
     *   dL / dC is the change in loss with respect to C, which should have been computed already
     */

    const float* A = &self->buffer[a->offset];
    const float* B = &self->buffer[b->offset];
    const float* dC = &self->grad_buffer[c->offset];
    float* dA = &self->grad_buffer[a->offset];
    float* dB = &self->grad_buffer[b->offset];

    for (uint32_t i = 0; i < m * n; i++) {
      dA[i] = 0.0f;
    }

    for (uint32_t i = 0; i < n * p; i++) {
      dB[i] = 0.0f;
    }

    /* Compute dA = dC * B ^ T */
    for (uint32_t i = 0; i < m; i++) {
      for (uint32_t j = 0; j < n; j++) {
        for (uint32_t k = 0; k < p; k++) {
          dA[i * n + j] += dC[i * p + k] * B[j * p + k];
        }
      }
    }

    /* Compute dB = A ^ T * dC */
    for (uint32_t i = 0; i < n; i++) {
      for (uint32_t j = 0; j < p; j++) {
        for (uint32_t k = 0; k < m; k++) {
          dB[i * p + j] += A[k * n + i] * dC[k * p + j];
        }
      }
    }

    return NANONET_OK;
  }

  NANONET_FUNC enum NanoNet_Status NanoNet_Back_ReLU(struct NanoNet_VM* self,
                                                     const struct NanoNet_Reg* a,
                                                     const struct NanoNet_Reg* c)
  {
    const uint16_t size = ((uint16_t)a->rows) * ((uint16_t)a->cols);

    const float* A = &self->buffer[a->offset];
    const float* dC = &self->grad_buffer[c->offset];
    float* dA = &self->grad_buffer[a->offset];

    for (uint16_t i = 0; i < size; i++) {
      dA[i] = (A[i] > 0.0f) ? dC[i] : 0.0f;
    }

    return NANONET_OK;
  }

  NANONET_FUNC enum NanoNet_Status NanoNet_Back_MSE(struct NanoNet_VM* self,
                                                    const struct NanoNet_Reg* a,
                                                    const struct NanoNet_Reg* b,
                                                    const struct NanoNet_Reg* c)
  {
    NANONET_UNUSED(c);

    /**
     * This function computes the change in loss with respect to the predicted matrix.
     *
     * dMSE / dY
     *
     * Which is equal to 2 / n (Y - Y_0)
     *
     * Where Y_0 is the ground truth.
     *
     * In the code, per convention, A is the first operand and is equal to the predicted matrix.
     * B is the second operand and equal to the ground truth matrix.
     */

    const uint16_t size = ((uint16_t)a->rows) * ((uint16_t)a->cols);

    const float inv = 1.0F / ((float)size);
    const float* A = &self->buffer[a->offset];
    const float* B = &self->buffer[b->offset];
    float* dA = &self->grad_buffer[a->offset];

    /**
     * Note: The predictions should always be the first operand and the ground truth is the second operand.
     *       No gradient is computed for the ground truth.
     */

    for (uint16_t i = 0; i < size; i++) {
      dA[i] = inv * (A[i] - B[i]);
    }

    return NANONET_OK;
  }

  NANONET_FUNC enum NanoNet_Status NanoNet_Backward(struct NanoNet_VM* self,
                                                    const uint32_t* opcodes,
                                                    uint32_t num_opcodes)
  {
    enum NanoNet_Status status = NANONET_OK;

    for (uint32_t i = num_opcodes; i > 0; i--) {

      const uint32_t opcode = opcodes[i - 1];
      const uint8_t op = (opcode >> 24) & 0xff;

      const struct NanoNet_Reg* a = &self->regs[(opcode >> 16) & 0xff];
      const struct NanoNet_Reg* b = &self->regs[(opcode >> 8) & 0xff];
      const struct NanoNet_Reg* c = &self->regs[opcode & 0xff];

      switch ((enum NanoNet_Op)op) {
        case NANONET_OP_MATMUL:
          status = NanoNet_Back_MatMul(self, a, b, c);
          break;
        case NANONET_OP_ADD:
          break;
        case NANONET_OP_MUL:
          break;
        case NANONET_OP_ROWCAT:
          break;
        case NANONET_OP_COLCAT:
          break;
        case NANONET_OP_MSE:
          status = NanoNet_Back_MSE(self, a, b, c);
          break;
        case NANONET_OP_SIGMOID:
          break;
        case NANONET_OP_RELU:
          status = NanoNet_Back_ReLU(self, a, c);
          break;
        case NANONET_OP_TANH:
          break;
        default:
          return NANONET_BAD_OPCODE;
      }

      if (status != NANONET_OK) {
        break;
      }
    }

    return status;
  }

  NANONET_FUNC void NanoNet_GradientDescent(struct NanoNet_VM* self,
                                            const uint8_t reg,
                                            const float learning_rate,
                                            float* weights)
  {
    struct NanoNet_Reg* r = &self->regs[reg];

    const uint16_t size = ((uint16_t)r->rows) * ((uint16_t)r->cols);

    for (uint16_t i = 0; i < size; i++) {
      const float g = self->grad_buffer[r->offset + i];
      const float x = self->buffer[r->offset + i];
      weights[i] = x + g * learning_rate;
    }
  }

#endif /* NANONET_INFERENCE_ONLY == 0 */

  NANONET_FUNC void NanoNet_Reset(struct NanoNet_VM* self)
  {
    self->buffer_offset = 0;
#if NANONET_INFERENCE_ONLY != 0
    self->grad_offset = 0;
#endif
  }

  NANONET_FUNC enum NanoNet_Status NanoNet_SetRegData(struct NanoNet_VM* self,
                                                      const uint8_t reg_index,
                                                      const uint8_t rows,
                                                      const uint8_t cols,
                                                      const float* buffer)
  {
    struct NanoNet_Reg* reg = &self->regs[reg_index];
    float* dst = NanoNet_AllocReg(self, reg, rows, cols);
    if (!dst) {
      return NANONET_NO_MEMORY;
    }
    memcpy(dst, buffer, ((size_t)rows) * ((size_t)cols) * sizeof(float));
    return NANONET_OK;
  }

  NANONET_FUNC const float* NanoNet_GetRegData(const struct NanoNet_VM* self, const uint8_t reg_index)
  {
    return &self->buffer[self->regs[reg_index].offset];
  }

#ifdef __cplusplus
} /* extern "C" { */
#endif

#endif /* NANONET_IMPLEMENTATION */
