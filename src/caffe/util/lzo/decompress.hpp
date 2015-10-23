/**
 * @brief Decompress LZO-compressed data
 *
 * @author Nikolaus Mayer, 2015 (mayern@cs.uni-freiburg.de)
 */

/// Local files
#include "minilzo.h"


namespace caffe {
namespace lzo {


class LZO_Decompressor {
  
public:
  
  /**
   * Constructor
   */
  LZO_Decompressor(unsigned int maximum_uncompressed_size)
    : m_maximum_uncompressed_size(maximum_uncompressed_size)
  {
    m_in  = new unsigned char __LZO_MMODEL[m_maximum_uncompressed_size];
    m_out = new unsigned char __LZO_MMODEL[m_maximum_uncompressed_size +
                                           m_maximum_uncompressed_size/16 +
                                           64 + 3];
    
    /// Initialize the LZO library
    /// TODO for every instance?
    if (lzo_init() != LZO_E_OK)
    {
      LOG(FATAL) << "internal error - lzo_init() failed !!!\n"
                 << "(this usually indicates a compiler bug - try recompiling\n"
                 << "without optimizations, and enable '-DLZO_DEBUG' for "
                 << "diagnostics)\n";
    }
  };
  
  
  /**
   * Destructor
   */
  ~LZO_Decompressor() 
  {
    if (m_in)  { delete[] m_in;  }
    if (m_out) { delete[] m_out; }
  };

  
  /**
   * Decompress LZO-compressed data
   */
  void Decompress(unsigned char* in_out,
                  unsigned int compressed_size)
  {
    int ret_val;
    lzo_uint inflated_size;
    
    lzo_memcpy(m_out, in_out, compressed_size);
    
    /// Decompress data
    r = lzo1x_decompress(out, compressed_size, m_in, &inflated_size, NULL);
    if (r != LZO_E_OK) {
      LOG(FATAL) << "internal error - decompression failed: " << r;
    }
    
    /// Retrieve inflated data
    lzo_memcpy(in_out, m_in, inflated_size);
  };
  
protected:
//   lzo_align_t __LZO_MMODEL m_wrkmem[((LZO1X_1_MEM_COMPRESS)+(sizeof(lzo_align_t)-1))
//                                   /sizeof(lzo_align_t)];
  unsigned int m_maximum_uncompressed_size;
  unsigned char* __LZO_MMODEL m_in;
  unsigned char* __LZO_MMODEL m_out;
  
};



}  // namespace lzo
}  // namespace caffe

