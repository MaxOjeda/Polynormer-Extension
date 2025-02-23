from model import Polynormer, PolynormerGraph

def parse_method(args, n, c, d, edge_attr_dim, device, task='node'):
    if task == 'node':
        model = Polynormer(d, args.hidden_channels, c, local_layers=args.local_layers, global_layers=args.global_layers,
                           in_dropout=args.in_dropout, dropout=args.dropout, global_dropout=args.global_dropout,
                           heads=args.num_heads, beta=args.beta, pre_ln=args.pre_ln, edge_dim=edge_attr_dim).to(device)
    elif task == 'graph':
        model = PolynormerGraph(d, args.hidden_channels, c,
                                local_layers=args.local_layers, global_layers=args.global_layers, batch_norm=args.batch_norm,
                                in_dropout=args.in_dropout, dropout=args.dropout, global_dropout=args.global_dropout,
                                heads=args.num_heads, beta=args.beta, pre_ln=args.pre_ln, edge_dim=edge_attr_dim, use_edges=args.use_edges).to(device)
    else:
        raise ValueError("Unknown task type")
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local_epochs', type=int, default=1000,
                        help='warmup epochs for local attention')
    parser.add_argument('--global_epochs', type=int, default=1000,
                        help='epochs for local-to-global attention')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')

    # model
    parser.add_argument('--method', type=str, default='poly')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--local_layers', type=int, default=7,
                        help='number of layers for local attention')
    parser.add_argument('--global_layers', type=int, default=2,
                        help='number of layers for global attention')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--beta', type=float, default=-1.0,
                        help='Polynormer beta initialization')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--use_edges', type=int, default=0)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global_dropout', type=float, default=None)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')
