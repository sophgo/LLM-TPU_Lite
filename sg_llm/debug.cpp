typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 10; // mantissa
    uint16_t exp : 5;   // exponent
    uint16_t sign : 1;  // sign
  } format;
} fp16;

typedef union {
  float fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp : 8;   // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

#if defined(__INTEL_COMPILER)
#include <immintrin.h>
#endif

#if defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
#include <intrin.h>
#endif

static inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
  return _CopyFloatFromInt32((__int32)w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

static inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
  return (uint32_t)_CopyInt32FromFloat(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

static inline float fp16_ieee_to_fp32_value(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||              \
    defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructud single-precision
   * number by 2**(-24), i.e. the same ammount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

float f16_to_f32(uint16_t src) {
  fp16 half = {.bits = src};
  if (half.format.exp == 31 && half.format.frac != 0) {
    fp32 res = {0};
    // NAN which had beed checked with IC
    res.bits = UINT32_C(0xFFC00000);
    return res.fval;
  }

  return fp16_ieee_to_fp32_value(src);
}

static inline uint32_t bf16_to_fp32_bits(uint16_t h) {
  // BF16 的位模式是：1 位符号，8 位指数，7 位尾数
  // 我们需要将其转换为 float 的位模式：1 位符号，8 位指数，23 位尾数
  // 扩展 BF16 到 32 位，尾数部分需要填充 16 位的 0
  uint32_t sign = (uint32_t)(h & 0x8000) << 16; // 符号位
  uint32_t exp  = (uint32_t)(h & 0x7F80) << 16; // 指数位
  uint32_t frac = (uint32_t)(h & 0x007F) << 16; // 尾数位

  // 将尾数的 7 位左移，以对齐到 23 位尾数的位置
  // frac <<= (23 - 7);

  // 组合成 float 的位模式
  return sign | exp | frac;
}

void dump_f16_tensor(bm_handle_t bm_handle, bm_device_mem_t mem,
                     const std::string &tip) {
  std::vector<uint16_t> data(10);
  bm_memcpy_d2s_partial(bm_handle, data.data(), mem, data.size() * 2);
  std::cout << ">>>>>>>>>>>>>>>>>>>>>" << tip << std::endl;
  // 打印前 10 个数据
  for (int i = 0; i < 10; i++) {
    std::cout << f16_to_f32(data[i]) << " ";
  }
  std::cout << "\n<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
}

void dump_bf16_tensor(bm_handle_t bm_handle, bm_device_mem_t mem,
                      const std::string &tip) {
  std::vector<uint16_t> data(10);
  bm_memcpy_d2s_partial(bm_handle, data.data(), mem, data.size() * 2);
  std::cout << ">>>>>>>>>>>>>>>>>>>>>" << tip << std::endl;
  fp32 t;
  // 打印前 10 个数据
  for (int i = 0; i < 10; i++) {
    t.bits = bf16_to_fp32_bits(data[i]);
    std::cout << t.fval << std::endl;
  }
  std::cout << "-------------------------------------" << std::endl;
}