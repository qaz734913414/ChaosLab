#include <core/core.hpp>
#include "version.hpp"

namespace chaos
{
	std::string GetMXVI()
	{
		std::stringstream ss;
		ss << "MxNet Extension for ChaosCV " << RELEASE_VER_STR << " [MSC v." << _MSC_VER << " " << __T(_MSC_PLATFORM_TARGET) << " bit]";
		return ss.str();
	}
}