deps = [
]

if subproject('seticore').found()
    seticore_subproj = subproject(
        'seticore',
        default_options: [
            'default_library='+get_option('default_library')
        ]
    )

    deps += declare_dependency(
        dependencies: seticore_subproj.get_variable('deps'),
        include_directories: seticore_subproj.get_variable('incs'),
        link_with: seticore_subproj.get_variable('libseticore'),
    )
endif

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('BLADE_LOADER_SETICORE', true)
    dep_lst += deps
endif

ldr_lst += {'Seticore': all_deps_found}
