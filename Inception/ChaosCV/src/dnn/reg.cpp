#include "dnn/reg.hpp"

namespace chaos
{
	namespace dnn
	{
		std::vector<Framework> Registered::frameworks = std::vector<Framework>();
		Framework& Registered::Have(const std::string& name)
		{
			return *std::find_if(frameworks.begin(), frameworks.end(),
				[=](const Framework& f) { return f.name == name; });
		}

		Framework& Register(const std::string& name)
		{
			// If already registered, return it directly
			auto it = std::find_if(Registered::frameworks.begin(), Registered::frameworks.end(),
				[=](const Framework& f) { return f.name == name; });

			if (Registered::frameworks.end() == it)
			{
				Registered::frameworks.push_back(Framework(name));
				return Registered::frameworks.back();
			}
			return *it;
		}
	}
}