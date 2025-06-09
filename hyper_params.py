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
    "dataset": "magazine",
    "item_id": None,         # Disable use of item metadata
    "category_id": None,     # Disable use of category metadata
    "use_gini": False,       # Avoid need to compute item-level stats
    "float64": False,
    "depth": 1,
    "grid_search_lamda": True,
    "lamda": 1.0,
    "user_support": -1,
    "seed": 42,
}
