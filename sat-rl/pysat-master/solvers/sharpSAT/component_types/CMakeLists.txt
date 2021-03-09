

SET(component_types_SOURCES 
	base_packed_component.cpp 
	component_archetype.cpp
)

ADD_LIBRARY(component_types MODULE ${component_types_SOURCES})
# target_include_directories(component_types PUBLIC ../..)