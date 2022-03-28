def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    import autoseg

    import autoseg.commands.sketch_communities
    import autoseg.commands.expand_communities
    import autoseg.commands.id_blocks

    modules = [autoseg.commands.sketch_communities,
               autoseg.commands.expand_communities,
               autoseg.commands.id_blocks,
               ]

    subparsers = parser.add_subparsers(title='Choose a command')
    subparsers.required = 'True'

    def get_str_name(mod):
        return os.path.splitext(os.path.basename(mod.__file__))[0]

    for module in modules:
        this_parser = subparsers.add_parser(get_str_name(module), description=module.__doc__)
        module.add_args(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
