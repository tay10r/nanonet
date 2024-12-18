#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <NanoNet.h>

#include <array>
#include <random>

#include <iostream>

namespace {

class training_loop
{
public:
  virtual ~training_loop() { free(vm_); }

  void run(const int steps_per_epoch, const int num_training_epochs, const float learning_rate)
  {
    nn_reset(vm_);

    test_loss_.emplace_back(test());

    for (auto i = 0; i < num_training_epochs; i++) {
      nn_reset(vm_);
      train(learning_rate);

      nn_reset(vm_);
      test_loss_.emplace_back(test());
    }
  }

  [[nodiscard]] auto get_test_loss() -> std::vector<float> { return test_loss_; }

protected:
  virtual void train(float learning_rate) = 0;

  virtual auto test() -> float = 0;

  [[nodiscard]] auto vm() -> nn_vm* { return vm_; }

  template<int Rows, int Cols>
  void set_reg(const uint8_t reg_index, const Eigen::Matrix<float, Rows, Cols>& m)
  {
    nn_set_reg(vm_, reg_index, Rows, Cols, m.data());
  }

  [[nodiscard]] auto uniform_int(int min_v, int max_v) -> int
  {
    std::uniform_int_distribution<int> dist(min_v, max_v);
    return dist(rng_);
  }

  [[nodiscard]] auto uniform_float(float min_v, float max_v) -> float
  {
    std::uniform_real_distribution<float> dist(min_v, max_v);
    return dist(rng_);
  }

private:
  std::mt19937 rng_{ 0 };

  nn_vm* vm_{ nn_vm_new() };

  std::vector<float> test_loss_;
};

} // namespace

//==========//
// XOR Test //
//==========//

namespace {

class xor_training_loop final : public training_loop
{
public:
protected:
  [[nodiscard]] auto test() -> float override
  {
    forward_pass();
    const float loss = *nn_get_reg(vm(), 8);
    return loss;
  }

  void train(const float learning_rate) override
  {
    forward_pass();

    // backward pass
    EXPECT_EQ(nn_backward(vm(), loss_code_.data(), loss_code_.size()), NANONET_OK);
    EXPECT_EQ(nn_backward(vm(), xor_code_.data(), xor_code_.size()), NANONET_OK);

    // update weights
    nn_gradient_descent(vm(), /*reg=*/0, learning_rate, weights_0_);
    nn_gradient_descent(vm(), /*reg=*/4, learning_rate, weights_1_);
  }

  void forward_pass()
  {
    //
    ASSERT_EQ(nn_set_reg(vm(), /*reg_index=*/0, 2, 2, weights_0_), NANONET_OK);
    ASSERT_EQ(nn_set_reg(vm(), /*reg_index=*/4, 1, 2, weights_1_), NANONET_OK);

    const int in[2]{ uniform_int(0, 1), uniform_int(0, 1) };
    const float real_in[2]{ static_cast<float>(in[0]), static_cast<float>(in[1]) };
    ASSERT_EQ(nn_set_reg(vm(), /*reg_index=*/1, 2, 1, real_in), NANONET_OK);

    EXPECT_EQ(nn_forward(vm(), xor_code_.data(), xor_code_.size()), NANONET_OK);

    // forward pass (compute loss)
    const int expected{ in[0] ^ in[1] };
    const float expected_real{ static_cast<float>(expected) };
    ASSERT_EQ(nn_set_reg(vm(), /*reg_index=*/7, 1, 1, real_in), NANONET_OK);

    EXPECT_EQ(nn_forward(vm(), loss_code_.data(), loss_code_.size()), NANONET_OK);
  }

private:
  float weights_0_[4]{ uniform_float(-1, 1), uniform_float(-1, 1), uniform_float(-1, 1), uniform_float(-1, 1) };

  float weights_1_[2]{ uniform_float(-1, 1), uniform_float(-1, 1) };

  const std::array<uint32_t, 4> xor_code_{
    0x01'00'01'02, // mat_mul %0 %1 -> %2
    0x42'02'00'03, //    relu %2    -> %3
    0x01'04'03'05, // mat_mul %4 %3 -> %5
    0x42'05'00'06  //    relu %5    -> %6
  };

  const std::array<uint32_t, 1> loss_code_{
    0x21'06'07'08 // mse_loss %6 %7 -> %8
  };
};

} // namespace

TEST(Train, Xor)
{
  constexpr auto epochs{ 32 };
  constexpr auto steps_per_epoch{ 16 };
  constexpr auto learning_rate{ 0.01F };

  xor_training_loop xor_trainer;
  xor_trainer.run(/*steps_per_epoch=*/16, /*num_training_epochs=*/32, /*learning_rate=*/0.01F);
  const auto test_loss = xor_trainer.get_test_loss();
  EXPECT_NEAR(test_loss.front(), 0.9F, 0.1F);
  EXPECT_NEAR(test_loss.back(), 0.0F, 0.1F);
}
