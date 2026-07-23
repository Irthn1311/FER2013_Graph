import copy, pytest, torch
from d16.scripts.validate_ofix7_mid_limit_audit import factory_semantics
from d16.scripts.run_ofix7_mid_limit_variant import validate_cell_semantics
from d16.scripts import prepare_ofix7_mid_limit_audit as prep
from d16.training import train_d16 as trainer

def test_s1_scheduler_semantics():
    data=factory_semantics()
    assert data["s1_optimizer_class"]=="AdamW"
    assert data["s1_scheduler_class"]=="CosineAnnealingLR"
    assert data["s1_T_max"]==90 and data["s1_eta_min"]==3e-5
    assert data["s1_step_calls"]==[[]]

def test_o1_radam_decoupled_semantics():
    data=factory_semantics()
    assert data["o1_optimizer_class"]=="RAdam"
    assert data["o1_decoupled_weight_decay"] is True
    assert data["o1_scheduler_class"]=="ReduceLROnPlateau"
    assert data["o1_step_calls"]==[[1.2]]

def test_unknown_optimizer_and_scheduler_rejected():
    p=torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError): trainer._build_optimizer([p],{"optimizer":{"type":"mystery"}})
    opt=torch.optim.AdamW([p])
    with pytest.raises(ValueError): trainer._build_scheduler(opt,{"scheduler":{"type":"mystery"}},90,1)

def test_combined_radam_cosine_rejected_by_registered_cell_guard():
    cfg=prep.load_yaml(prep.config_path("O1",42))
    combined=copy.deepcopy(cfg);combined["training"]["scheduler"]={"type":"cosine","t_max":90,"eta_min":3e-5}
    with pytest.raises(RuntimeError): validate_cell_semantics("O1",combined)

