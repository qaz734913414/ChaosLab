#pragma once

#include "dnn/tensor.hpp"
#include "utils/json.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Inputs
		{
		public:
			//Inputs();
			int node_id;
			int index;
			int version;
		};

		class CHAOS_API Symbol
		{
		public:
			Symbol();
			Symbol(Json json);

			std::string op;
			std::string name;
			std::map<std::string, std::string> attrs;
			std::vector<Inputs> inputs;
		};

		using SymbolList = std::vector<Symbol>;

		SymbolList Load(const Json& json);
	}
}