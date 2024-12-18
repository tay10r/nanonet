#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <NanoNet.h>

namespace {

template<typename Func>
auto
test_binary_op(uint8_t op, Func func) -> nn_status
{
  const Eigen::Matrix4f a = Eigen::Matrix4f::Random();
  const Eigen::Matrix4f b = Eigen::Matrix4f::Random();
  // $op$ %0x00 %0x01 -> %0x02
  const uint32_t opcode = (((uint32_t)op) << 24) | 0x00000102;
  auto vm = nn_vm_new();
  nn_set_reg(vm, /*reg_index=*/0, 4, 4, a.data());
  nn_set_reg(vm, /*reg_index=*/1, 4, 4, b.data());
  const auto status = nn_forward(vm, &opcode, 1);
  const Eigen::Matrix4f result(nn_get_reg(vm, 2));
  func(a, b, result);
  free(vm);
  return status;
}

} // namespace

TEST(Operators, MatMul)
{
  // matmul %0x00 %0x01 -> %0x02
  const uint32_t opcode = 0x01000102;

  Eigen::Matrix4f a = Eigen::Matrix4f::Random();
  Eigen::Matrix4f b = Eigen::Matrix4f::Random();
  auto vm = nn_vm_new();
  nn_set_reg(vm, /*reg_index=*/0, 4, 4, a.data());
  nn_set_reg(vm, /*reg_index=*/1, 4, 4, b.data());
  const auto status = nn_forward(vm, &opcode, 1);
  EXPECT_EQ(status, NANONET_OK);
  Eigen::Map<const Eigen::Matrix4f> result(nn_get_reg(vm, 2));
  EXPECT_EQ((a * b).isApprox(result), true);
  free(vm);
}

TEST(Operators, Add)
{
  const auto status =
    test_binary_op(0x02, [](const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, const Eigen::Matrix4f& result) {
      EXPECT_EQ((a + b).isApprox(result), true);
    });
  EXPECT_EQ(status, NANONET_OK);
}

TEST(Operators, Mul)
{
  const auto status =
    test_binary_op(0x03, [](const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, const Eigen::Matrix4f& result) {
      EXPECT_EQ((a.cwiseProduct(b)).isApprox(result), true);
    });
  EXPECT_EQ(status, NANONET_OK);
}

namespace {

template<typename Func>
auto
test_unary_op(uint8_t op, Func func) -> nn_status
{
  const Eigen::Matrix4f a = Eigen::Matrix4f::Random();
  const uint32_t opcode = (((uint32_t)op) << 24) | 0x00000001;
  auto vm = nn_vm_new();
  nn_set_reg(vm, /*reg_index=*/0, 4, 4, a.data());
  const auto status = nn_forward(vm, &opcode, 1);
  const Eigen::Matrix4f result(nn_get_reg(vm, 1));
  func(a, result);
  free(vm);
  return status;
}

} // namespace

TEST(Operators, ReLU)
{
  const auto status = test_unary_op(0x42, [](const Eigen::Matrix4f& input, const Eigen::Matrix4f& result) {
    EXPECT_EQ(input.cwiseMax(0.0F).isApprox(result), true);
  });
  EXPECT_EQ(status, NANONET_OK);
}
