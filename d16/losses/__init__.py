"""D16 loss components."""
from d16.losses.hard_proto_separation import (
    HardPrototypeSeparationLoss,
    build_hard_proto_separation_loss,
    hard_proto_lambda,
)
from d16.losses.main_logit_pair_margin import (
    MainLogitPairMarginLoss,
    build_main_logit_pair_margin_loss,
    main_logit_pair_margin_lambda,
)
from d16.losses.pairwise_hard_relation import (
    PairwiseHardRelationLoss,
    build_pairwise_hard_relation_loss,
    pairwise_hard_relation_lambda,
)
