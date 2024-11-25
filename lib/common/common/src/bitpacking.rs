/// For performance reasons, both [`BitWriter`] and [`BitReader`] use an
/// intermediate buffer. Instead of writing/reading a single byte at a time,
/// they write/read `size_of::<Buf>()` bytes at once.
/// This is an implementation detail and shouldn't affect the data layout. Any
/// unsigned numeric type larger than `u32` should work.
type Buf = u64;

/// Writes bits to `u8` vector.
/// It's like [`std::io::Write`], but for bits rather than bytes.
pub struct BitWriter<'a> {
    output: &'a mut Vec<u8>,
    buf: Buf,
    buf_bits: u8,
}

impl<'a> BitWriter<'a> {
    #[inline]
    pub fn new(output: &'a mut Vec<u8>) -> Self {
        output.clear();
        Self {
            output,
            buf: 0,
            buf_bits: 0,
        }
    }

    /// Write a `value` of `bits` bits to the output.
    ///
    /// The `bits` must be less than or equal to 32, and the `value` must fit in
    /// the `bits` bits.
    #[inline]
    pub fn write(&mut self, value: u32, bits: u8) {
        #[cfg(test)]
        debug_assert!(bits <= 32 && u64::from(value) < 1 << bits);

        self.buf |= Buf::from(value) << self.buf_bits;
        self.buf_bits += bits;
        if self.buf_bits >= Buf::BITS as u8 {
            // 32 bits         0    64  buf_bits                    0
            // |   |           |    |      |                        |
            //  ....cccccvvvvvvv     .......bbbbbbbbbbbbbbbbbbbbbbbbb
            //             value                     initial self.buf
            //
            //
            //           bits+buf_bits
            //                 |    64   buf_bits                   0
            //                 |    |      |                        |
            //                  cccccvvvvvvvbbbbbbbbbbbbbbbbbbbbbbbbb
            //                  └[2]┘└─────────────[1]──────────────┘
            //                            self.buf and value combined
            //
            // [1]: this part is already written to the output
            // [2]: we need to put the remaining bits to the buffer
            self.output.extend_from_slice(&self.buf.to_le_bytes());
            self.buf_bits -= Buf::BITS as u8;
            self.buf = Buf::from(value) >> (bits - self.buf_bits);
        }
    }

    /// Write the remaining bufferized bits to the output.
    #[inline]
    pub fn finish(self) {
        self.output.extend_from_slice(
            &self.buf.to_le_bytes()[..(self.buf_bits as usize).div_ceil(u8::BITS as usize)],
        );
    }
}

/// Reads bits from `u8` slice.
/// It's like [`std::io::Read`], but for bits rather than bytes.
pub struct BitReader<'a> {
    input: &'a [u8],
    buf: Buf,
    buf_bits: u8,
    mask: Buf,
    bits: u8,
}

impl<'a> BitReader<'a> {
    #[inline]
    pub fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            buf: 0,
            buf_bits: 0,
            mask: 0,
            bits: 0,
        }
    }

    /// Configure the reader to read `bits` bits at a time. The subsequent calls
    /// to [`BitReader::read()`] will read `bits` bits at a time.
    #[inline]
    pub fn set_bits(&mut self, bits: u8) {
        self.bits = bits;
        self.mask = (1 << bits) - 1;
    }

    /// Read next `bits` bits from the input. The amount of bits must be set
    /// with [`BitReader::set_bits()`] before calling this method.
    ///
    /// If read beyond the end of the input, the result would be an unspecified
    /// garbage.
    #[inline]
    pub fn read(&mut self) -> u32 {
        if self.buf_bits < self.bits {
            let new_bits = read_buf_and_advance(&mut self.input);
            let val = ((self.buf | (new_bits << self.buf_bits)) & self.mask) as u32;
            self.buf = new_bits >> (self.bits - self.buf_bits);
            self.buf_bits += Buf::BITS as u8 - self.bits;
            val
        } else {
            self.buf_bits -= self.bits;
            let val = (self.buf & self.mask) as u32;
            self.buf >>= self.bits;
            val
        }
    }
}

/// Read a single [`Buf`] from the `input` and advance (or not) the `input`.
#[inline]
fn read_buf_and_advance(input: &mut &[u8]) -> Buf {
    let mut buf = 0;
    if input.len() >= size_of::<Buf>() {
        // This line translates to a single unaligned pointer read.
        buf = Buf::from_le_bytes(input[0..size_of::<Buf>()].try_into().unwrap());
        // This line translates to a single pointer advance.
        *input = &input[size_of::<Buf>()..];
    } else {
        // We could remove this branch by explicitly using unsafe pointer
        // operations in the branch above, but we are playing it safe here.
        for (i, byte) in input.iter().copied().enumerate() {
            buf |= Buf::from(byte) << (i * u8::BITS as usize);
        }

        // Don't advance the input for performance reasons: this should be the
        // last read.
        // *input = &[]; // Not needed.
    }
    buf
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use rand::rngs::StdRng;
    use rand::{Rng as _, SeedableRng as _};

    use super::*;

    #[test]
    fn test_bitpacking() {
        let mut rng = StdRng::seed_from_u64(42);

        let mut bits_per_value = Vec::new();
        let mut values = Vec::new();
        let mut packed = Vec::new();
        let mut unpacked = Vec::new();
        for len in 0..40 {
            for _ in 0..100 {
                values.clear();
                bits_per_value.clear();
                let mut total_bits = 0;
                for _ in 0..len {
                    let bits = rng.gen_range(0u8..=32u8);
                    values.push(rng.gen_range(0..(1u64 << bits)) as u32);
                    bits_per_value.push(bits);
                    total_bits += u64::from(bits);
                }

                let mut w = BitWriter::new(&mut packed);
                for (&x, &bits) in zip(&values, &bits_per_value) {
                    w.write(x, bits);
                }
                w.finish();

                assert_eq!(packed.len(), total_bits.next_multiple_of(8) as usize / 8);

                unpacked.clear();
                let mut r = BitReader::new(&packed);
                for &bits in &bits_per_value {
                    r.set_bits(bits);
                    unpacked.push(r.read());
                }

                assert_eq!(values, unpacked);
            }
        }
    }
}
