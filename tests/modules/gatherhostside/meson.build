if cfg_lst.get('BLADE_MODULE_GATHERHOSTSIDE', false)
    if py.found()
        # ATA-like accumulation.
        test(
            'test-modules-gatherhostside-ata',
            py,
            args: [files('gatherhostside.py'), '2', '192', '8192', '2', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # VLA-like accumulation.
        test(
            'test-modules-gatherhostside-vla',
            py,
            args: [files('gatherhostside.py'), '27', '131072', '3', '2', '2', '16'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        # Miscellaneous configurations.

        test(
            'test-modules-gatherhostside-misc-1',
            py,
            args: [files('gatherhostside.py'), '3', '7', '3', '1', '0', '5'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherhostside-misc-2',
            py,
            args: [files('gatherhostside.py'), '1', '2', '5', '6', '1', '6'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherhostside-misc-3',
            py,
            args: [files('gatherhostside.py'), '11', '1', '13', '9', '2', '7'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherhostside-misc-4',
            py,
            args: [files('gatherhostside.py'), '9', '2', '1', '3', '3', '8'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )

        test(
            'test-modules-gatherhostside-misc-5',
            py,
            args: [files('gatherhostside.py'), '9', '2', '5555', '3', '2', '8'],
            is_parallel: false,
            timeout: 0,
            env: 'PYTHONPATH=@0@'.format(py_build_path),
        )
    endif
endif
