from sbatchpy.run import run

run(
    {
        "job-name": f"run.sh",
        "output": f"run.out",
        "mem": "200Gb",
        "time": "20:00:00",
        "account": "sua183",
        "cpus-per-task": "10",
        "partition": "shared",
        # "partition": "shared",
        "ntasks-per-node": "1",
        "nodes": "1",

    },
    code=f'date \n source activate test \n python -u rmc_fcc.py  \n date ',
)
