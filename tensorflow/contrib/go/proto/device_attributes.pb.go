// Code generated by protoc-gen-go.
// source: tensorflow/core/framework/device_attributes.proto
// DO NOT EDIT!

package tensorflow

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// BusAdjacency identifies the ability of a device to participate in
// maximally efficient DMA operations within the local context of a
// process.
//
// This is currently ignored.
type BusAdjacency int32

const (
	BusAdjacency_BUS_0               BusAdjacency = 0
	BusAdjacency_BUS_1               BusAdjacency = 1
	BusAdjacency_BUS_ANY             BusAdjacency = 2
	BusAdjacency_BUS_NUM_ADJACENCIES BusAdjacency = 3
)

var BusAdjacency_name = map[int32]string{
	0: "BUS_0",
	1: "BUS_1",
	2: "BUS_ANY",
	3: "BUS_NUM_ADJACENCIES",
}
var BusAdjacency_value = map[string]int32{
	"BUS_0":               0,
	"BUS_1":               1,
	"BUS_ANY":             2,
	"BUS_NUM_ADJACENCIES": 3,
}

func (x BusAdjacency) String() string {
	return proto.EnumName(BusAdjacency_name, int32(x))
}
func (BusAdjacency) EnumDescriptor() ([]byte, []int) { return fileDescriptor2, []int{0} }

type DeviceAttributes struct {
	Name string `protobuf:"bytes,1,opt,name=name" json:"name,omitempty"`
	// String representation of device_type.
	DeviceType string `protobuf:"bytes,2,opt,name=device_type,json=deviceType" json:"device_type,omitempty"`
	// Memory capacity of device in bytes.
	MemoryLimit  int64        `protobuf:"varint,4,opt,name=memory_limit,json=memoryLimit" json:"memory_limit,omitempty"`
	BusAdjacency BusAdjacency `protobuf:"varint,5,opt,name=bus_adjacency,json=busAdjacency,enum=tensorflow.BusAdjacency" json:"bus_adjacency,omitempty"`
	// A device is assigned a global unique number each time it is
	// initialized. "incarnation" should never be 0.
	Incarnation uint64 `protobuf:"fixed64,6,opt,name=incarnation" json:"incarnation,omitempty"`
	// String representation of the physical device that this device maps to.
	PhysicalDeviceDesc string `protobuf:"bytes,7,opt,name=physical_device_desc,json=physicalDeviceDesc" json:"physical_device_desc,omitempty"`
}

func (m *DeviceAttributes) Reset()                    { *m = DeviceAttributes{} }
func (m *DeviceAttributes) String() string            { return proto.CompactTextString(m) }
func (*DeviceAttributes) ProtoMessage()               {}
func (*DeviceAttributes) Descriptor() ([]byte, []int) { return fileDescriptor2, []int{0} }

func init() {
	proto.RegisterType((*DeviceAttributes)(nil), "tensorflow.DeviceAttributes")
	proto.RegisterEnum("tensorflow.BusAdjacency", BusAdjacency_name, BusAdjacency_value)
}

var fileDescriptor2 = []byte{
	// 317 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0x5c, 0x51, 0x4b, 0x4f, 0xf3, 0x30,
	0x10, 0xfc, 0xd2, 0xa7, 0xba, 0xe9, 0x87, 0xa2, 0x05, 0x81, 0x6f, 0x14, 0x4e, 0x88, 0x43, 0x1f,
	0xc0, 0x95, 0x43, 0xd2, 0xf6, 0x40, 0x05, 0x55, 0x95, 0xd2, 0x03, 0xa7, 0xc8, 0x71, 0x5d, 0x30,
	0x34, 0x71, 0x65, 0xbb, 0x54, 0xf9, 0xf1, 0x48, 0x24, 0xe9, 0x2b, 0xe2, 0xb6, 0x3b, 0x33, 0xbb,
	0x9e, 0xf5, 0x40, 0xcf, 0xf0, 0x58, 0x4b, 0xb5, 0x58, 0xca, 0x4d, 0x87, 0x49, 0xc5, 0x3b, 0x0b,
	0x45, 0x23, 0xbe, 0x91, 0xea, 0xab, 0x33, 0xe7, 0xdf, 0x82, 0xf1, 0x80, 0x1a, 0xa3, 0x44, 0xb8,
	0x36, 0x5c, 0xb7, 0x57, 0x4a, 0x1a, 0x89, 0x70, 0x1c, 0xb9, 0xfe, 0xb1, 0xc0, 0x19, 0xe4, 0x3a,
	0xf7, 0x20, 0x43, 0x84, 0x4a, 0x9c, 0x2e, 0x21, 0x56, 0xcb, 0xba, 0x69, 0xf8, 0x79, 0x8d, 0x97,
	0x60, 0xef, 0xf6, 0x99, 0x64, 0xc5, 0x49, 0x29, 0xa7, 0x60, 0x0b, 0xbd, 0xa6, 0x08, 0x5e, 0x41,
	0x33, 0xe2, 0x91, 0x54, 0x49, 0xb0, 0x14, 0x91, 0x30, 0xa4, 0x92, 0x2a, 0xca, 0xbe, 0xbd, 0xc5,
	0x9e, 0x33, 0x08, 0x1f, 0xe1, 0x7f, 0xb8, 0xd6, 0x01, 0x9d, 0x7f, 0x52, 0xc6, 0x63, 0x96, 0x90,
	0x6a, 0xaa, 0x39, 0xb9, 0x23, 0xed, 0xa3, 0xa1, 0xb6, 0xb7, 0xd6, 0xee, 0x9e, 0xf7, 0x9b, 0x61,
	0xa1, 0xc3, 0x16, 0xd8, 0x22, 0x66, 0x54, 0xc5, 0xd4, 0x08, 0x19, 0x93, 0x5a, 0x3a, 0x5c, 0xf3,
	0x8b, 0x10, 0x76, 0xe1, 0x6c, 0xf5, 0x91, 0x68, 0xc1, 0xe8, 0x32, 0xd8, 0xb9, 0x9d, 0x73, 0xcd,
	0x48, 0x3d, 0x77, 0x8b, 0x7b, 0x6e, 0x7b, 0xf0, 0x20, 0x65, 0x6e, 0x47, 0xd0, 0x2c, 0xbe, 0x88,
	0x0d, 0xa8, 0x7a, 0xb3, 0x69, 0xd0, 0x75, 0xfe, 0xed, 0xcb, 0x9e, 0x63, 0xa1, 0x0d, 0xf5, 0xac,
	0x74, 0xc7, 0x6f, 0x4e, 0x09, 0x2f, 0xe0, 0x34, 0x6b, 0xc6, 0xb3, 0x97, 0xc0, 0x1d, 0x8c, 0xdc,
	0xfe, 0x70, 0xdc, 0x7f, 0x1a, 0x4e, 0x9d, 0xb2, 0xf7, 0x00, 0x44, 0xaa, 0xf7, 0xe2, 0x31, 0x87,
	0x2c, 0xbc, 0xf3, 0xbf, 0x9f, 0x3c, 0xc9, 0xa2, 0xd0, 0x13, 0x2b, 0xac, 0xe5, 0xa1, 0xdc, 0xff,
	0x06, 0x00, 0x00, 0xff, 0xff, 0xa9, 0x59, 0xe4, 0x98, 0xc9, 0x01, 0x00, 0x00,
}