# These were the settings the authors put in comment themselves.
# hyper_params = {
#     "dataset": "ml-1m",
#     "item_id": "item_id:token",  # configure it based on the .item file
#     "category_id": "genre:token_seq",  # configure it based on the .item file
#     "use_gini": True,
#     "float64": False,
#     "depth": 1,
#     "grid_search_lamda": True,
#     "lamda": 1.0,  # Only used if grid_search_lamda == False
#     # Number of users to keep (randomly)
#     "user_support": -1,  # -1 implies use all users
#     "seed": 42,
# }

# These were the settings Max put in comment
#hyper_params = {
#    "dataset": "douban",
#    "item_id": "id:token",  # configure it based on the .item file
#    "category_id": "publisher:token",  # configure it based on the .item file
#    "use_gini": True,
#    "float64": False,
#    "depth": 1,
#    "grid_search_lamda": True,
#    "lamda": 1.0,  # Only used if grid_search_lamda == False
#    # Number of users to keep (randomly)
#    "user_support": -1,  # -1 implies use all users
#    "seed": 42,
#}

hyper_params = {
    "dataset": "douban",            # Dataset name
    "item_id": "item_id:token",         # Disable use of item metadata
    "category_id": "genre:token_seq",     # Disable use of category metadata
    "use_gini": False,# Avoid need to compute item-level stats
    "use_unfairness_gap": True,  # Avoid need to compute unfairness gap
    "evaluate_random": False,  # Evaluate random model 
    "float64": False,
    "depth": 1,
    "grid_search_lamda": True,
    "lamda": 1.0,
    "user_support": -1,
    "seed": 42,
    "grid_search_alpha": False,  # Enable grid search for alpha
    "alpha": 20.0,  # Default alpha value, used if grid_search
    "use_exposure_aware_reranking": False,  # Enable exposure aware reranking
    "use_user_group_fairness_reranking": False,  # Enable user-group fairness reranking
}

