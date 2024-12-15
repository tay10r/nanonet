#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <NanoNet.h>

typedef uint32_t opcode_t;

static inline opcode_t
enc_matmul(uint8_t op1, uint8_t op2, uint8_t dst)
{
  return 0x01u | (op1 << 8) | (op2 << 16) | (dst << 24);
}

static inline opcode_t
enc_add(uint8_t op1, uint8_t op2, uint8_t dst)
{
  return 0x02u | (op1 << 8) | (op2 << 16) | (dst << 24);
}

static inline opcode_t
enc_mul(uint8_t op1, uint8_t op2, uint8_t dst)
{
  return 0x03u | (op1 << 8) | (op2 << 16) | (dst << 24);
}

static inline opcode_t
enc_sigmoid(uint8_t src, uint8_t dst)
{
  return 0x41 | (src << 8) | (dst << 24);
}

struct reg
{
  uint8_t rows;
  uint8_t cols;
  uint16_t offset;
};

struct vm
{
  struct reg regs[256];

  uint8_t num_inputs;

  float buffer[65536];

  uint16_t buffer_offset;

  float grad_buffer[65536];

  uint16_t grad_buffer_offset;
};

enum vm_status
{
  VM_OK,
  VM_SHAPE_ERROR,
  VM_BAD_OPCODE,
  VM_BUFFER_OVERFLOW
};

static inline uint16_t
vm_buffer_remaining(const struct vm* self)
{
  return 0xffffU - self->buffer_offset;
}

static inline float*
vm_alloc(struct vm* self, uint16_t s)
{
  if (vm_buffer_remaining(self) < s) {
    return NULL;
  }
  float* ptr = &self->buffer[self->buffer_offset];
  self->buffer_offset += s;
  return ptr;
}

static inline enum vm_status
vm_exec_matmul(struct vm* self, uint8_t op1, uint8_t op2, uint8_t dst)
{
  const struct reg* o1 = &self->regs[op1];
  const struct reg* o2 = &self->regs[op2];
  struct reg* d = &self->regs[dst];

  if (o1->cols != o2->rows) {
    return VM_SHAPE_ERROR;
  }

  d->rows = o1->rows;
  d->cols = o2->cols;

  uint16_t s = ((uint16_t)d->rows) * ((uint16_t)d->cols);

  float* dst_buffer = vm_alloc(self, s);
  if (!dst_buffer) {
    return VM_BUFFER_OVERFLOW;
  }

  const float* o1_buf = &self->buffer[o1->offset];
  const float* o2_buf = &self->buffer[o2->offset];

  for (uint8_t i = 0; i < o1->rows; i++) {
    for (uint8_t j = 0; j < o2->cols; j++) {
      float sum = 0.0F;
      for (uint8_t k = 0; k < o1->cols; k++) {
        sum += o1_buf[i * o1->cols + k] * o2_buf[k * o2->cols + j];
      }
      dst_buffer[i * d->cols + j] = sum;
    }
  }

  return VM_OK;
}

static inline enum vm_status
vm_exec_add(struct vm* self, uint8_t op1, uint8_t op2, uint8_t dst)
{
  const struct reg* o1 = &self->regs[op1];
  const struct reg* o2 = &self->regs[op2];
  struct reg* d = &self->regs[dst];

  if ((o1->rows != o2->rows) || (o1->cols != o2->cols)) {
    return VM_SHAPE_ERROR;
  }

  d->rows = o1->rows;
  d->cols = o1->cols;

  uint16_t s = ((uint16_t)o1->rows) * ((uint16_t)o1->cols);

  float* dst_buffer = vm_alloc(self, s);
  if (!dst_buffer) {
    return VM_BUFFER_OVERFLOW;
  }

  for (uint16_t i = 0; i < s; i++) {
    dst_buffer[i] = self->buffer[o1->offset + i] + self->buffer[o2->offset + i];
  }

  return VM_OK;
}

static inline enum vm_status
vm_exec_mul(struct vm* self, uint8_t op1, uint8_t op2, uint8_t dst)
{
  const struct reg* o1 = &self->regs[op1];
  const struct reg* o2 = &self->regs[op2];
  struct reg* d = &self->regs[dst];

  if ((o1->rows != o2->rows) || (o1->cols != o2->cols)) {
    return VM_SHAPE_ERROR;
  }

  d->rows = o1->rows;
  d->cols = o1->cols;

  uint16_t s = ((uint16_t)o1->rows) * ((uint16_t)o1->cols);

  float* dst_buffer = vm_alloc(self, s);
  if (!dst_buffer) {
    return VM_BUFFER_OVERFLOW;
  }

  for (uint16_t i = 0; i < s; i++) {
    dst_buffer[i] = self->buffer[o1->offset + i] * self->buffer[o2->offset + i];
  }

  return VM_OK;
}

static inline enum vm_status
vm_exec_sigmoid(struct vm* self, uint8_t src, uint8_t dst)
{
  const struct reg* s = &self->regs[src];

  struct reg* d = &self->regs[dst];

  d->rows = s->rows;
  d->cols = s->cols;

  const uint16_t size = ((uint16_t)s->rows) * ((uint16_t)s->cols);

  float* dst_buffer = vm_alloc(self, size);
  if (!dst_buffer) {
    return VM_BUFFER_OVERFLOW;
  }

  for (uint16_t i = 0; i < size; i++) {
    const float x = self->buffer[s->offset + i];
    dst_buffer[i] = 1.0F / (1.0F + expf(-x));
  }

  return VM_OK;
}

static inline enum vm_status
vm_exec(struct vm* self, const opcode_t op)
{
  uint8_t op1 = (op >> 8) & 0xff;
  uint8_t op2 = (op >> 16) & 0xff;
  uint8_t dst = (op >> 24) & 0xff;
  switch (op & 0xff) {
    case 0x01:
      return vm_exec_matmul(self, op1, op2, dst);
    case 0x02:
      return vm_exec_add(self, op1, op2, dst);
    case 0x03:
      return vm_exec_mul(self, op1, op2, dst);
    case 0x41:
      return vm_exec_sigmoid(self, op1, dst);
  }
  return VM_BAD_OPCODE;
}

enum vm_status
vm_forward(struct vm* self, const uint32_t num_opcodes, const opcode_t* ops)
{
  for (uint32_t i = 0; i < num_opcodes; i++) {
    enum vm_status s = vm_exec(self, ops[i]);
    printf("%d: %d\n", i, ops[i] & 0xff);
    assert(s == VM_OK);
    if (s != VM_OK) {
      return s;
    }
  }
  return VM_OK;
}

void
vm_reset(struct vm* self)
{
  memset(self, 0, sizeof(struct vm));
}

enum vm_status
vm_push_mat(struct vm* self, const uint8_t rows, const uint8_t cols, const float* data)
{
  const uint16_t size = ((uint16_t)rows) * ((uint16_t)cols);

  if (vm_buffer_remaining(self) < size) {
    return VM_BUFFER_OVERFLOW;
  }

  memcpy(&self->buffer[self->buffer_offset], data, ((size_t)size) * sizeof(float));

  struct reg* r = &self->regs[self->num_inputs];
  r->rows = rows;
  r->cols = cols;
  r->offset = self->buffer_offset;

  self->buffer_offset += size;

  self->num_inputs++;

  return VM_OK;
}

int
main()
{
  // clang-format off

  // there are 4 inputs
  const opcode_t m[] = {
    enc_matmul(1, 0, 4),
    enc_add(2, 4, 5),
    enc_matmul(3, 5, 6),
    enc_sigmoid(6, 7)
  };
  const float op1[4] = { 1, 2, 3, 4 };
  const float op2[16] = { 5, 6, 7, 8,
                          9, 1, 2, 3,
                          4, 5, 6, 7,
                          8, 9, 1, 2 };
  const float op3[4] = { 3, 4, 5, 6 };
  const float op4[4] = { 7, 8, 9, 1 };
  // clang-format on

  struct vm v;
  vm_reset(&v);
  vm_push_mat(&v, 4, 1, op1);
  vm_push_mat(&v, 4, 4, op2);
  vm_push_mat(&v, 4, 1, op3);
  vm_push_mat(&v, 1, 4, op3);
  assert(v.num_inputs == 4);

  const enum vm_status status = vm_forward(&v, sizeof(m) / sizeof(m[0]), m);
  assert(status == VM_OK);

  assert(v.regs[4].rows == 4);
  assert(v.regs[4].cols == 1);

  assert(v.regs[7].rows == 1);
  assert(v.regs[7].cols == 1);

  const float result = v.buffer[v.regs[7].offset];

  printf("result: %f\n", result);

  return 0;
}
