#include "caffe/util/binarydb.hpp"

#include <string>

#include <boost/python.hpp>

namespace bp = boost::python;

namespace caffe { namespace db {

template <typename Dtype>
void BinaryDB<Dtype>::Open(const string& source, const LayerParameter& param) {
  top_num_ = param.top_size();
  sample_variants_num_ = param.data_param().sample_size();
  
  LOG(INFO) << "Opening BinaryDB using boost::python";
  
  string param_str;
  param.SerializeToString(&param_str);
  
  Py_Initialize();
  try {
    bp::object module = bp::import("binarydb");
    bp::object dbclass = module.attr("BinaryDB")(param_str, top_num_);
    
    bp::list infos = (bp::list)dbclass.attr("getInfos")();
    
//     int n = bp::len((ret));
//     
//     std::cout << "Len: " << n << std::endl;
//     for(unsigned int i=0; i<n; i++){
//       string str = boost::python::extract<string>((ret)[i]);
//       std::cout << "String: " << str << std::endl;
//     }
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
  LOG(INFO) << "Opened BinaryDB " << source;
}


template <typename Dtype>
void BinaryDB<Dtype>::Close() {
  //TODO
}

template <typename Dtype>
int BinaryDB<Dtype>::get_num_samples() {
  return 0; // TODO
}

template <typename Dtype>
void BinaryDB<Dtype>::get_sample(int index, vector<Blob<Dtype>*>* dst) {
  //TODO
}

const unsigned char* srcptr=(const unsigned char*)datum.data().c_str();
    Dtype* destptr=ptr;

//     int channel_start = -1; //inclusive
//     int channel_end = 0; //non-inclusive (end will become start in next slice)
//     for(int slice = 0; slice <= slice_points.size(); slice++)
//     {
//         channel_start = channel_end;
// 
//         if(slice == slice_points.size())
//             channel_end = channels;
//         else
//             channel_end = slice_points[slice];
// 
//         int channel_count=channel_end-channel_start;
// 
//         int format;
//         if(encoding.size()<=slice)
//             format=DataParameter_CHANNELENCODING_UINT8;
//         else
//             format=encoding[slice];
// 
// //         LOG(INFO) << "Slice " << slice << "(" << channel_start << "," << channel_end << ") has format " << ((int)format);
//         switch(format)
//         {
//             case DataParameter_CHANNELENCODING_UINT8:
//                 for(int c=0; c<channel_count; c++)
//                     for(int y=0; y<height; y++)
//                         for(int x=0; x<width; x++)
//                             *(destptr++)=static_cast<Dtype>(*(srcptr++));
//                 break;
//             case DataParameter_CHANNELENCODING_UINT16FLOW:
//             for(int c=0; c<channel_count; c++)
//                 for(int y=0; y<height; y++)
//                     for(int x=0; x<width; x++)
//                     {
//                         short v;
//                         *((unsigned char*)&v)=*(srcptr++);
//                         *((unsigned char*)&v+1)=*(srcptr++);
// 
//                         Dtype value;
//                         if(v==std::numeric_limits<short>::max()) {
//                           value = std::numeric_limits<Dtype>::signaling_NaN();
//                         } else {
//                           value = ((Dtype)v)/32.0;
//                         }
// 
//                         *(destptr++)=value;
//                     }
//                 break;
//             case DataParameter_CHANNELENCODING_BOOL1:
//                 {
//                     int j=0;
//                     for(int i=0; i<(width*height-1)/8+1; i++)
//                     {
//                         unsigned char data=*(srcptr++);
//                         for(int k=0; k<8; k++)
//                         {
//                             float value=(data&(1<<k))==(1<<k);
//                             if(j<width*height)
//                                 *(destptr++)=value?1.0:0;
//                             j++;
//                         }
//                     }
//                 }
//                 break;
//             default:
//                 LOG(FATAL) << "Invalid format for slice " << slice;
//                 break;
//         }
//     }
// //     LOG(INFO) << destptr << " " << ptr;
//     assert(destptr==ptr+count);


INSTANTIATE_CLASS(BinaryDB);

}  // namespace db
}  // namespace caffe
