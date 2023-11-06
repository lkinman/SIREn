def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    import siren

    import siren.commands.sketch_communities
    import siren.commands.expand_communities
    import siren.commands.preprocess
    import siren.commands.fine_tune
    import siren.commands.train
    import siren.commands.eval_model

    modules = [siren.commands.sketch_communities,
               siren.commands.expand_communities,
               siren.commands.preprocess,
               siren.commands.fine_tune,
               siren.commands.train,
               siren.commands.eval_model
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
