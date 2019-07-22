#include "dnn/layers/data_layer.hpp"

namespace chaos
{
	namespace dnn
	{
		DataLayer::DataLayer() : name(std::string()), shape(Shape()) {}
		DataLayer::DataLayer(const std::string& name, const Shape& shape) : name(name), shape(shape) {}
	}
}