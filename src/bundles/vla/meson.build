#
#  Mode S
#

deps = [
    'BLADE_MODULE_SETICORE'
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

sum_lst += {'[VLA] Mode S': all_deps_found}

cfg_lst.set('BLADE_BUNDLE_VLA_MODE_S', all_deps_found)