#include "dnn/net.hpp"

namespace chaos
{
	namespace dnn
	{
		Context::Context() {};
		Context::Context(const DeviceType& type, int id) : type(type), id(id) {};

		Model::Model() {}
		Model::Model(const std::string& weight) : symbol(std::string()), weight(weight) {}
		Model::Model(const std::string& symbol, const std::string& weight) : symbol(symbol), weight(weight) {}

		Net::~Net() {}
	} // namespace dnn
} // namespace chaos