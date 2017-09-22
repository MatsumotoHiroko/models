require 'csv'
require 'rmagick'
require 'fileutils'
 
SIZE = 32
BIN_DIR = './data/cifar-10-batches-bin/'
 
if(ARGV[0] == 'train')
  csv_data = CSV.read('train.csv', headers: true)
else
  csv_data = CSV.read('eval.csv', headers: true)
end
puts "start..."
 
file = String.new
csv_data.each do |data|
  buf = String.new
  puts data
  buf << [data[1].to_i()].pack('C')
  img = Magick::Image.read(data[0]).first.resize(SIZE,SIZE)
  %w(red green blue).each do |color|
   img.each_pixel do |px|
    buf << [px.send(color) >> 8].pack('C')
   end
  end
  file << buf
end
 
FileUtils.mkdir_p(BIN_DIR) unless FileTest.exist?(BIN_DIR)
if(ARGV[0] == 'train')
  File.binwrite(BIN_DIR + "data_batch_1.bin", file)
  File.binwrite(BIN_DIR + "data_batch_2.bin", file)
  File.binwrite(BIN_DIR + "data_batch_3.bin", file)
  File.binwrite(BIN_DIR + "data_batch_4.bin", file)
  File.binwrite(BIN_DIR + "data_batch_5.bin", file)
  File.binwrite(BIN_DIR + "data_batch_6.bin", file)
else
  File.binwrite(BIN_DIR + "test_batch.bin", file)
end