import sys
import os
import caffe.proto.caffe_pb2 as cpb
import google.protobuf as pb
from google.protobuf import text_format

class BinaryDB:

    class SampleProps:
        pass

    def __init__(self, param_str, num_top_blobs):
        # Get the LayerParameters
        param = cpb.LayerParameter()
        param.ParseFromString(param_str)
        #print(param)

        self.data_param = param.data_param

        self.min_margin = param.data_param.minimum_boundary_margin

        # Read collection list
        assert(os.path.isfile(self.data_param.collection_list))

        with open(self.data_param.collection_list) as f:
            self.clips = [line.rstrip('\n') for line in f if len(line.rstrip('\n')) > 0]

        print("Clips: ")
        print(self.clips)

        # Dimensions for each entry
        self.named_entry_format_dimensions = {}
        self.output_entries_dimensions = None

        self.bin_filenames = []

        # Set up sample offset ranges
        self.sample_props = []
        self.num_entries_per_sample = None
        self.setUpSamples()

        # Find valid samples in all clips
        self.findSamplesInClips()

        #print(self.all_samples)

    def throw_error(self, str):
        raise AssertionError(str)

    def setUpSamples(self):

        for sample in self.data_param.sample:
            min_offset = None
            max_offset = None
            needed_entries = set()

            if self.num_entries_per_sample is None:
                self.num_entries_per_sample = len(sample.entry)
            else:
                if self.num_entries_per_sample != len(sample.entry):
                    self.throw_error('Inconsistent number of entries per sample')

            for entry in sample.entry:
                needed_entries.add(entry.name)
                if min_offset is None or min_offset > entry.offset:
                    min_offset = entry.offset
                if max_offset is None or max_offset < entry.offset:
                    max_offset = entry.offset

            self.sample_props.append(BinaryDB.SampleProps())
            self.sample_props[-1].min_offset = min_offset
            self.sample_props[-1].max_offset = max_offset
            self.sample_props[-1].needed_entries = needed_entries.copy()


    def setOrCheckDimensions(self, entry_name, dimensions):
        if entry_name in self.named_entry_format_dimensions:
            if self.named_entry_format_dimensions[entry_name] != dimensions:
                self.throw_error('Dimensions not consistent for entry %s' % entry_name)
        else:
            self.named_entry_format_dimensions[entry_name] = dimensions


    def enc_size(self, encoding):
        if encoding == cpb.BinaryDB.UINT8:
            return 1
        elif encoding == cpb.BinaryDB.FIXED16DIV32:
            return 2

    def filenameIndex(self, list, item):
        utf_str = str(item).encode('utf-8')
        if utf_str not in list:
            list.append(utf_str)

        return list.index(utf_str)

    def findSamplesInClips(self):

        self.all_samples = []
        for clip_folder in self.clips:
            index_file = os.path.join(self.data_param.source, clip_folder, "index.prototxt")
            print("Index File: " + index_file)
            assert(os.path.isfile(index_file))
            with open(index_file) as f:
                index_str = f.read()

            bindb_index = cpb.BinaryDB()
            text_format.Merge(index_str, bindb_index)
            #print(bindb_index)

            slice_points = [int(s) for s in bindb_index.slice_before]
            #print("Slice points:")
            #print(slice_points)

            for sampidx, sample in enumerate(self.data_param.sample):
                entry_lookup = {}
                #print("== Considering sample %d" % sampidx)
                # Search all available BIN files mentioned in this index
                # for the needed entries and store where to find what
                for binfile in bindb_index.file:
                    #print("Scanning entry_formats for binfile %s" % binfile.filename)
                    for entry_format_idx, entry_format in enumerate(binfile.content.entry_format):
                        if entry_format.name in self.sample_props[sampidx].needed_entries:
                            # Create lookup entry
                            if entry_format.name in entry_lookup:
                                self.throw_error('Entry %s defined multiple times in %s' % (entry_format.name, index_file))
                            #print(" Adding entry_format %s" % entry_format.name)
                            entry_lookup[entry_format.name] = {
                                'filename': binfile.filename,
                                'encoding': entry_format.data_encoding,
                                'dimensions': (entry_format.width, entry_format.height, entry_format.channels),
                                'index': entry_format_idx,
                                'num': len(binfile.content.entry_format)
                            }
                # Now for current sample definition find all samples in this CLIP/DB
                num_total = bindb_index.num
                start_off = max(-self.sample_props[sampidx].min_offset, self.min_margin)
                end_off = min(-self.sample_props[sampidx].max_offset, -self.min_margin)
                #print("Num_total: %d" % num_total)
                #print("Start/End off: %d/%d" % (start_off, end_off))

                for index in range(start_off, num_total+end_off):
                    for slice_point in slice_points:
                        if slice_point + end_off <= index < slice_point + start_off:
                            break
                    else:
                        entries = []
                        for entry in sample.entry:
                            if entry.name not in entry_lookup:
                                self.throw_error('The requested entry %s is not available in clip %s.\nAvailable: %s' % (entry.name, clip_folder, str(entry_lookup)))
                            el = entry_lookup[entry.name]
                            dims = el['dimensions']

                            abs_file_path = os.path.join(self.data_param.source, clip_folder, el['filename'])
                            if not os.path.isfile(abs_file_path):
                                self.throw_error('Bin file %s from clip %s does not exist' % (abs_file_path, clip_folder))
                            bin_file_index = self.filenameIndex(self.bin_filenames, abs_file_path)
                            encoding = el['encoding']
                            self.setOrCheckDimensions(entry.name, el['dimensions'])

                            entry_index = (index + entry.offset) * el['num'] + el['index']
                            byte_offset = (dims[0]*dims[1]*dims[2]*self.enc_size(encoding)+4)*entry_index
                            entries.append((bin_file_index, byte_offset, encoding))

                        self.all_samples.append(entries)

                # Store dimensions for all entries of this sample (Later will check if samples are compatible)
                entries_dimensions = [entry_lookup[entry.name]['dimensions'] for entry in sample.entry]
                if self.output_entries_dimensions is None:
                    self.output_entries_dimensions = entries_dimensions
                else:
                    for i in range(len(entries_dimensions)):
                        #print(entries_dimensions)
                        #print(self.output_entries_dimensions)
                        if entries_dimensions[i] != self.output_entries_dimensions[i]:
                            self.throw_error('Dimensions of sample %d not consistent over clips (because of entry %d)' % (sampidx, i))

        print 'Number of samples: %d' % len(self.all_samples)
        sys.stdout.flush()


    def getInfos(self):
        if self.num_entries_per_sample != len(self.output_entries_dimensions):
            self.throw_error('Num of entries per sample is not consistent')

        return (self.all_samples, self.output_entries_dimensions, self.bin_filenames)
