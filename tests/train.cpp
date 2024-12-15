#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <NanoNet.h>

#include <random>

#include <iostream>

TEST(Train, Xor)
{
  std::mt19937 rng(/*seed=*/0);
  std::uniform_int_distribution<int> in_dist(0, 1);
  std::uniform_real_distribution<float> w_dist(-1.0F, 1.0F);

  const uint32_t opcodes_xor[4]{
    0x01'00'01'02, // mat_mul %0 %1 -> %2
    0x42'02'00'03, //    relu %2    -> %3
    0x01'04'03'05, // mat_mul %4 %3 -> %5
    0x42'05'00'06  //    relu %5    -> %6
  };

  const uint32_t opcodes_loss{
    0x21'06'07'08 // mse_loss %6 %7 -> %8
  };

  constexpr auto epochs{ 8 };
  constexpr auto steps_per_epoch{ 32 };
  constexpr auto learning_rate{ 0.01F };

  auto vm = NanoNet_New();
  float w0[4]{ w_dist(rng), w_dist(rng), w_dist(rng), w_dist(rng) };
  float w1[2]{ w_dist(rng), w_dist(rng) };
  for (auto e = 0; e < epochs; e++) {
    float train_loss_sum{ 0.0F };
    for (auto i = 0; i < steps_per_epoch; i++) {
      NanoNet_Reset(vm);
      ASSERT_EQ(NanoNet_SetRegData(vm, /*reg_index=*/0, 2, 2, w0), NANONET_OK);
      ASSERT_EQ(NanoNet_SetRegData(vm, /*reg_index=*/4, 2, 1, w1), NANONET_OK);
      // forward pass (compute xor)
      const int in[2]{ in_dist(rng), in_dist(rng) };
      const float real_in[2]{ static_cast<float>(in[0]), static_cast<float>(in[1]) };
      ASSERT_EQ(NanoNet_SetRegData(vm, /*reg_index=*/1, 2, 1, real_in), NANONET_OK);
      EXPECT_EQ(NanoNet_Forward(vm, opcodes_xor, 4), NANONET_OK);
      // forward pass (compute loss)
      const int expected{ in[0] ^ in[1] };
      const float expected_real{ static_cast<float>(expected) };
      ASSERT_EQ(NanoNet_SetRegData(vm, /*reg_index=*/5, 1, 1, real_in), NANONET_OK);
      EXPECT_EQ(NanoNet_Forward(vm, &opcodes_loss, 1), NANONET_OK);
      const float loss = *NanoNet_GetRegData(vm, 8);
      train_loss_sum += loss;
      // backward pass
      EXPECT_EQ(NanoNet_Backward(vm, &opcodes_loss, 1), NANONET_OK);
      EXPECT_EQ(NanoNet_Backward(vm, opcodes_xor, 4), NANONET_OK);
      // update weights
      NanoNet_GradientDescent(vm, /*reg=*/0, learning_rate, w0);
      NanoNet_GradientDescent(vm, /*reg=*/4, learning_rate, w1);
    }
    const auto train_loss_avg = train_loss_sum / static_cast<float>(steps_per_epoch);
    std::cout << "loss: " << train_loss_avg;
  }

  NanoNet_Free(vm);
}
