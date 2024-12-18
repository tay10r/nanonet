#include <gtest/gtest.h>

#include <NanoNet.h>

#include <sstream>

namespace {

auto
print_module(const nn_module* m) -> std::string;

} // namespace

namespace {

NANONET_DEF_MODULE(empty_module)
{
  return NANONET_INPUT(1, 1);
}

} // namespace

TEST(Module, EmptyModule)
{
  nn_module* mod{ nullptr };
  const auto status = nn_build_module(&mod, empty_module, nullptr, nullptr);
  EXPECT_EQ(status, NANONET_OK);
  const auto str = print_module(mod);
  free(mod);
  EXPECT_EQ(str, "");
}

namespace {

NANONET_DEF_MODULE(simple_module)
{
  auto x = NANONET_INPUT(4, 1);
  x = NANONET_MATMUL(NANONET_PARAM(4, 4), x);
  x = NANONET_ADD(NANONET_PARAM(4, 1), x);
  x = NANONET_MATMUL(NANONET_PARAM(1, 4), x);
  return x;
}

} // namespace

TEST(Module, SimpleModule)
{
  nn_module* mod{ nullptr };
  const auto status = nn_build_module(&mod, simple_module, nullptr, nullptr);
  EXPECT_EQ(status, NANONET_OK);
  const auto str = print_module(mod);
  free(mod);
  EXPECT_EQ(str,
            "matmul %1 %0 -> %2\n"
            "add %3 %2 -> %4\n"
            "matmul %5 %4 -> %6\n");
}

namespace {

auto
opcode_to_string(const uint32_t op) -> std::string
{
  std::ostringstream s;
  auto unary{ false };
  switch ((op >> 24) & 0xff) {
    case 0x01:
      s << "matmul";
      break;
    case 0x02:
      s << "add";
      break;
    case 0x03:
      s << "mul";
      break;
    case 0x04:
      s << "rowcat";
      break;
    case 0x05:
      s << "colcat";
      break;
    case 0x21:
      s << "mse";
      break;
    case 0x41:
      s << "sigmoid";
      unary = true;
      break;
    case 0x42:
      s << "relu";
      unary = true;
      break;
    case 0x43:
      s << "tanh";
      unary = true;
      break;
  }
  s << " %";
  s << static_cast<int>((op >> 16) & 0xff);
  if (!unary) {
    s << " %";
    s << static_cast<int>((op >> 8) & 0xff);
  }
  s << " -> %";
  s << static_cast<int>(op & 0xff);
  return s.str();
}

auto
print_module(const nn_module* m) -> std::string
{
  std::ostringstream s;
  uint32_t num_opcodes{};
  const auto* opcodes = nn_get_module_code(m, &num_opcodes);
  for (uint32_t i = 0; i < num_opcodes; i++) {
    s << opcode_to_string(opcodes[i]) << '\n';
  }
  return s.str();
}

} // namespace
