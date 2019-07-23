#include "dnn/net.hpp"
#include "dnn/reg.hpp"

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
		Ptr<Net> Net::Load(const Model& model, const Context& ctx)
		{
			CHECK(model.from_file) << "General load funcion just support load net from file.";

			File symbol(model.symbol);
			File weight(model.weight);

			for (auto framework : Registered::frameworks)
			{
				if (framework.symbol_type == symbol.Type && framework.weight_type == weight.Type)
				{
					return framework.load_func(model, ctx);
				}
			}
			LOG(FATAL) << "Unknown inference framework for ." << symbol.Type << " and ." << weight.Type << " files.";
			return Ptr<Net>(); // Never reachable
		}

		Framework::Framework() {}
		Framework::Framework(const std::string& name) : name(name) {}
		Framework& Framework::With(const std::string& sym_type, const std::string& wgt_type, const LoadFunction& func)
		{
			symbol_type = sym_type;
			weight_type = wgt_type;
			load_func = func;

			return *this;
		}
	} // namespace dnn
} // namespace chaos