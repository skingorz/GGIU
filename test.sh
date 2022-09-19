#!/bin/bash
{
    METHOD=PN
    TASK=5shot
    TASKID=PN_codebase
    WANDB_PROJECT=FSLdoing
    model=epoch=55-step=13999.ckpt      # ckpt file name
    CONFIG_DIR=config
    CONFIG_FILE=config_test.yaml
    CONFIG_PY=config/set_config_PN_TTA_test.py # python file to set config

    git_status="$(git status 2> /dev/null)"
    branch_pattern="^(# )?On branch ([^${IFS}]*)"
    if [[ ${git_status} =~ ${branch_pattern} ]]; then
        BRANCH=${BASH_REMATCH[2]}
    fi

    TASK_Dir=results/${BRANCH}/${METHOD}/${TASK}/${WANDB_PROJECT}/${TASKID}

    mkdir -p ${TASK_Dir}

    ckpt_dir=${TASK_Dir}/checkpoints
    ckptPath=${ckpt_dir}/${model}

    # 1shot config
    mkdir ${TASK_Dir}/1shot_results
    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${ckptPath} 1
    cp ${CONFIG_DIR}/${CONFIG_FILE} ${TASK_Dir}/1shot_results
    cp test.sh ${TASK_Dir}/1shot_results

    # 5shot config
    mkdir ${TASK_Dir}/5shot_results
    python ${CONFIG_PY} ${METHOD} ${TASK} ${TASKID} ${WANDB_PROJECT} ${ckptPath} 5
    cp ${CONFIG_DIR}/${CONFIG_FILE} ${TASK_Dir}/5shot_results
    cp test.sh ${TASK_Dir}/5shot_results

    # test 1shot
    python run.py --config ${TASK_Dir}/1shot_results/${CONFIG_FILE}

    # test 5shot
    python run.py --config ${TASK_Dir}/5shot_results/${CONFIG_FILE}

    exit
}