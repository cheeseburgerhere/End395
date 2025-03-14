import pyomo.environ as pyo
import pandas as pd


#!pip install openpyxl

directory="Scenarios\ProjectPart1-Scenario1.xlsx"
#pandas depens on this package to read excel files
orders=pd.read_excel(directory,sheet_name="Orders")


trucks=pd.read_excel(directory,sheet_name="Vehicles")


pallets=pd.read_excel(directory,sheet_name="Pallets")


extraParam=pd.read_excel(directory,sheet_name="Parameters")



# --------------------------------------------------------
# 1) Create the model
# --------------------------------------------------------
model = pyo.ConcreteModel()


# --------------------------------------------------------
# 2) Define Sets
# --------------------------------------------------------
model.I = pyo.Set(initialize=range(1, len(pallets)+1), doc="Set of pallets")
model.K = pyo.Set(initialize=trucks["Vehicle Type"].unique().tolist(), doc="Set of vehicle types")

# Don't forget to change the indexes if there is a change in data, default order:
# 0	Planning Horizon (T)
# 1	Max. trips per period	
# 2	Max. pallets in area

model.T = pyo.Set(initialize=range(1, extraParam.iloc[0,1]+1), doc="Set of days")
model.M = pyo.Set(initialize=range(1, extraParam.iloc[1,1]+1), doc="Set of possible trips per day")

model.S = pyo.Set(initialize=range(1, pallets["Pallet Size"].max()+1), doc="Set of pallet sizes")
model.O = pyo.Set(initialize=orders["Order ID"].unique() ,doc="Set of orders")
# model.J = pyo.Set(initialize=range(1, int(sorted(pallets["Product Type"].unique())[-1][-1])+1), doc="Set of products") 
model.J = pyo.Set(initialize=pallets["Product Type"].unique(), doc="Set of products") 



capacity_dict = {}
for row in trucks["Vehicle Type"].unique():
    row=trucks[trucks["Vehicle Type"]==row].iloc[0,:]
    k=int(row.iloc[1])
    s=1
    capacity_dict[(k, s)] = row.iloc[2]
    k=int(row.iloc[1])
    s=2
    capacity_dict[(k, s)] = row.iloc[3]

demand_dict = {}
for row in orders.iterrows():
    o=row[1].iloc[0]
    j=row[1].iloc[1]
    demand_dict[(o, j)] = row[1].iloc[2]




# --------------------------------------------------------
# 3) Define Parameters
# --------------------------------------------------------
model.release_day = pyo.Param(model.I, initialize=dict(enumerate(pallets["Release Day"], start=1)), within=pyo.NonNegativeIntegers, doc="r_i: Release day for pallet i")
model.c_owned     = pyo.Param(model.K, initialize=dict(enumerate(trucks.groupby(["Vehicle Type","Fixed Cost (c_k)"]).count().reset_index().iloc[:,1], start=1)), within=pyo.NonNegativeReals, doc="c_k: Cost of using an owned vehicle type k")
model.c_rented    = pyo.Param(model.K, initialize=dict(enumerate(trucks.groupby(["Vehicle Type","Variable Cost (c'_k)"]).count().reset_index().iloc[:,1], start=1)), within=pyo.NonNegativeReals, doc="c'_k: Cost of renting vehicle type k")

#!!!!!!! Unchecked initialization
model.h           = pyo.Param(model.O, initialize=orders.groupby(["Order ID","Earliness Penalty"]).count().reset_index().iloc[:,0:2].set_index("Order ID").to_dict()["Earliness Penalty"], within=pyo.NonNegativeReals, doc="h_o: Per-day earliness penalty for order o")
model.d_due       = pyo.Param(model.O, initialize=orders.groupby(["Order ID","Due Date"]).count().reset_index().iloc[:,0:2].set_index("Order ID").to_dict()["Due Date"], within=pyo.NonNegativeIntegers, doc="d_o: Due day for order o")


model.p_demand    = pyo.Param(model.O, model.J, initialize=demand_dict, default=0, within=pyo.NonNegativeIntegers, doc="p_{o,j}: Demand for product j in order o")
demand_dict=None

model.j_of_i      = pyo.Param(model.I, initialize=dict(enumerate(pallets["Product Type"], start=1)), within=model.J, doc="j(i): Product type of pallet i")
model.n_i         = pyo.Param(model.I, initialize=dict(enumerate(pallets["Amount"], start=1)), within=pyo.NonNegativeIntegers, doc="n_i: Product capacity of pallet i")


model.capacity    = pyo.Param(model.K, model.S, initialize=capacity_dict, within=pyo.NonNegativeIntegers, doc="Capacity_{k,s}: # of pallets of size s that fit in vehicle k")
capacity_dict=None

model.b_k         = pyo.Param(model.K, initialize=dict(enumerate(trucks.groupby("Vehicle Type").count().reset_index().iloc[:,1], start=1)), within=pyo.NonNegativeIntegers, doc="b_k: Number of owned vehicles of type k available")

model.q           = pyo.Param(initialize=extraParam.iloc[2,1],within=pyo.NonNegativeIntegers, doc="q: Max number of pallets allowed in waiting area overnight")





# --------------------------------------------------------
# 4) Define Decision Variables
# --------------------------------------------------------
# 1) Shipment Variables
model.u = pyo.Var(model.K, model.T, model.M, model.S, within=pyo.Binary,
                  doc="u_{k,t,m,s}: 1 if a shipment of vehicle type k on day t, trip m, uses pallet size s")
model.v = pyo.Var(model.K, model.T, model.M, model.S, within=pyo.NonNegativeIntegers,
                  doc="v_{k,t,m,s}: Number of pallets of size s in shipment (k,t,m)")

# 2) Vehicle Usage
model.y = pyo.Var(model.K, model.T, model.M, within=pyo.Binary,
                  doc="y_{k,t,m}: 1 if an owned vehicle of type k is used on day t, trip m")
model.z = pyo.Var(model.K, model.T, model.M, within=pyo.Binary,
                  doc="z_{k,t,m}: 1 if a rented vehicle of type k is used on day t, trip m")

# 3) Pallet Assignment
model.x = pyo.Var(model.I, model.K, model.T, model.M, model.S, within=pyo.Binary,
                  doc="x_{i,k,t,m,s}: 1 if pallet i is shipped via (k,t,m,s)")

# 4) Order Fulfillment
model.e = pyo.Var(model.I, model.O, model.T, within=pyo.NonNegativeIntegers,
                  doc="e_{i,o,t}: products from pallet i allocated to order o and shipped on day t")


model.e_sub= pyo.Var(model.I, model.O, model.T, within=pyo.Binary,
                  doc="e_sub_{i,o,t}: if any products from pallet i allocated to order o and shipped on day t")



# After declaring model.x, fix variables where t < earliest_day[i]
fixedCount=0
for i in model.I:
    T_i = model.release_day[i]
    for t in model.T:
        if t < T_i:
            for k in model.K:
                for m in model.M:
                    for s in model.S:
                        model.x[i, k, t, m, s].fix(0)
                        fixedCount+=1

print(fixedCount, "variables fixed to 0")

fixedCountE=0
for i in model.I:
    T_i = model.release_day[i]
    for t in model.T:
        if t < T_i:
            for o in model.O:
                model.e[i, o, t].fix(0)
                model.e_sub[i, o, t].fix(0)
                fixedCountE+=1


print(fixedCountE*2, "e variables fixed to 0")
# --------------------------------------------------------
# 5) Define Constraints
# --------------------------------------------------------

# (1) Each pallet is shipped exactly once, after its release day
def single_shipment_rule(model, i):
    return sum(model.x[i, k, t, m, s]
               for k in model.K
               for t in model.T if t >= model.release_day[i]
               for m in model.M
               for s in model.S) == 1
model.single_shipment = pyo.Constraint(model.I, rule=single_shipment_rule)


# (2) Shipment Capacity:
#     sum_i x_{i,k,t,m,s} = v_{k,t,m,s} <= capacity_{k,s} * u_{k,t,m,s}
# We can express this in two separate constraints:
def capacity_link_1_rule(model, k, t, m, s):
    # sum of pallets assigned = v_{k,t,m,s}
    return sum(model.x[i, k, t, m, s] for i in model.I) == model.v[k, t, m, s]
model.capacity_link_1 = pyo.Constraint(model.K, model.T, model.M, model.S, rule=capacity_link_1_rule)

def capacity_link_2_rule(model, k, t, m, s):
    # v_{k,t,m,s} <= capacity_{k,s} * u_{k,t,m,s}
    return model.v[k, t, m, s] <= model.capacity[k, s] * model.u[k, t, m, s]
model.capacity_link_2 = pyo.Constraint(model.K, model.T, model.M, model.S, rule=capacity_link_2_rule)


# (3) Single pallet size per shipment
def single_size_rule(model, k, t, m):
    return sum(model.u[k, t, m, s] for s in model.S) <= 1
model.single_size = pyo.Constraint(model.K, model.T, model.M, rule=single_size_rule)


# (4) Vehicle Type usage constraint
#     y_{k,t,m} + z_{k,t,m} = sum_s u_{k,t,m,s}
def vehicle_type_rule(model, k, t, m):
    return model.y[k, t, m] + model.z[k, t, m] == sum(model.u[k, t, m, s] for s in model.S)
model.vehicle_type = pyo.Constraint(model.K, model.T, model.M, rule=vehicle_type_rule)


# (5) Owned Vehicle Daily Trip Limit
#     sum_m y_{k,t,m} <= 3 * b_k
# Here we assume each type k can do up to 3 trips per owned vehicle. Adapt if needed.
def owned_vehicle_limit_rule(model, k, t):
    return sum(model.y[k, t, m] for m in model.M) <= 3 * model.b_k[k]
model.owned_vehicle_limit = pyo.Constraint(model.K, model.T, rule=owned_vehicle_limit_rule)


# (6) Order Demand:
#     sum_{i: j(i)=j} sum_{t <= d_o} e_{i,o,t} >= p_{o,j}
# We iterate over each order o and each product j
# If p_demand[o,j] > 0, we must satisfy that demand by the due date.
def order_demand_rule(model, o, j):
    # sum of allocated products across all pallets i of type j
    return sum(model.e[i, o, t]
               for i in model.I
               if model.j_of_i[i] == j
               for t in model.T
               if t <= model.d_due[o]) >= model.p_demand[o, j]
model.order_demand = pyo.Constraint(model.O, model.J, rule=order_demand_rule)


# (7) Product allocation per pallet:
#     sum_o e_{i,o,t} <= n_i * sum_{k,m,s} x_{i,k,t,m,s}
def product_alloc_rule(model, i, t):
    return sum(model.e[i, o, t] for o in model.O) <= \
           model.n_i[i] * sum(model.x[i, k, t, m, s] for k in model.K for m in model.M for s in model.S)
model.product_alloc = pyo.Constraint(model.I, model.T, rule=product_alloc_rule)


# (8) Waiting Area Limit:
#     For each day t, the number of pallets that have been released but not shipped by day t cannot exceed q
#     sum_{i: r_i <= t} (1 - sum_{t' <= t, k,m,s} x_{i,k,t',m,s}) <= q
def waiting_area_rule(model, t):
    # Only consider pallets i whose release_day[i] <= t
    return sum(
        1 - sum(model.x[i, k, t_prime, m, s]
                for t_prime in model.T if t_prime <= t
                for k in model.K
                for m in model.M
                for s in model.S)
        for i in model.I
        if model.release_day[i] <= t
    ) <= model.q
model.waiting_area = pyo.Constraint(model.T, rule=waiting_area_rule)


M = 300

def e_sub_rule(model, i, o, t):
    # If any products from pallet i are allocated to order o on day t, then e_sub[i,o,t] must be 1.
    return model.e[i, o, t] <= M * model.e_sub[i, o, t]
model.e_sub_constraint = pyo.Constraint(model.I, model.O, model.T, rule=e_sub_rule)

# --------------------------------------------------------
# 6) Define Objective
# --------------------------------------------------------
# Minimize:
#   sum_{k,t,m} (c_k * y_{k,t,m} + c'_k * z_{k,t,m})
#   + sum_{i,o,t} (d_o - t)*h_o * e_{i,o,t}
def objective_rule(model):
    vehicle_cost = sum(model.c_owned[k]*model.y[k, t, m] + model.c_rented[k]*model.z[k, t, m]
                       for k in model.K for t in model.T for m in model.M)
    earliness_cost = sum((model.d_due[o] - t)*model.h[o]*model.e_sub[i, o, t]
                         for i in model.I
                         for o in model.O
                         for t in model.T
                         # typically you'd restrict to t <= d_due[o] if you only penalize earliness
                         if t <= model.d_due[o])
    return vehicle_cost + earliness_cost

model.Obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)



# --------------------------------------------------------
#Solver
solver= pyo.SolverFactory('gurobi')
solver.options['TimeLimit'] = 3600*6
solver.options["Threads"]=16


results = solver.solve(model, tee=True)
results.write()

# At this point, you can query solution values:
for i in model.I:
    for k in model.K:
        for t in model.T:
            for m in model.M:
                for s in model.S:
                    if pyo.value(model.x[i,k,t,m,s]) != 0:
                        print(f"Pallet {i} shipped by vehicle {k} on day {t}, trip {m}, size {s}")



print("\n")
# --------------------------------------------------------
# 7) Output
# At this point, you can query solution values:

for i in model.I:
    for o in model.O:
        for t in model.T:
                    if pyo.value(model.e[i,o,t]) != 0:
                        print(f"Pallet {i} shipped to order {o} on day {t}")
                        # print(f"Pallet {i} shipped by vehicle {k} on day {t}, trip {m}, size {s}")