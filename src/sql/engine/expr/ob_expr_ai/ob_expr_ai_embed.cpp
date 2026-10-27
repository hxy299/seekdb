/*
 * Copyright (c) 2025 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define USING_LOG_PREFIX SQL_ENG
#include "ob_expr_ai_embed.h"
#include "query/ai/ob_ai_endpoint_resolver.h"
#include "share/rc/ob_server_runtime.h"
#include "sql/engine/expr/ob_expr_lob_utils.h"

using namespace oceanbase::common;
using namespace oceanbase::sql;

namespace oceanbase 
{
namespace sql 
{

namespace
{
int build_dimension_config(common::ObIAllocator &allocator, const int64_t dimension, ObJsonObject *&config)
{
  int ret = OB_SUCCESS;
  config = NULL;
  if (dimension > 0) {
    ObJsonObject *config_obj = NULL;
    ObJsonInt *dim_json = NULL;
    if (OB_FAIL(ObAIFuncJsonUtils::get_json_object(allocator, config_obj))) {
    } else if (OB_FAIL(ObAIFuncJsonUtils::get_json_int(allocator, dimension, dim_json))) {
    } else if (OB_FAIL(config_obj->add("dimensions", dim_json))) {
    } else {
      config = config_obj;
    }
  }
  return ret;
}
} // namespace

ObExprAIEmbed::ObExprAIEmbed(common::ObIAllocator &alloc)
    : ObFuncExprOperator(alloc, 
                        T_FUN_SYS_AI_EMBED, 
                        N_AI_EMBED, 
                        MORE_THAN_ZERO,
                        NOT_VALID_FOR_GENERATED_COL, 
                        NOT_ROW_DIMENSION) 
{
}

ObExprAIEmbed::~ObExprAIEmbed() 
{
}

int ObExprAIEmbed::calc_result_typeN(ObExprResType &type,
                                     ObExprResType *types_stack,
                                     int64_t param_num,
                                     common::ObExprTypeCtx &type_ctx) const 
{
  UNUSED(type_ctx);
  UNUSED(types_stack);
  int ret = OB_SUCCESS;
  if (OB_UNLIKELY(param_num > 3 || param_num < 2)) {
    ObString func_name_(get_name());
    ret = OB_ERR_PARAM_SIZE;
    LOG_USER_ERROR(OB_ERR_PARAM_SIZE, func_name_.length(), func_name_.ptr());
  } else {
    types_stack[MODEL_IDX].set_calc_type(ObVarcharType);
    types_stack[MODEL_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    types_stack[CONTENT_IDX].set_calc_type(ObVarcharType);
    types_stack[CONTENT_IDX].set_calc_collation_type(CS_TYPE_UTF8MB4_BIN);
    if (param_num == 3) {
      if (ob_is_integer_type(types_stack[DIM_IDX].get_type())) {
        types_stack[DIM_IDX].set_calc_type(ObIntType);
        types_stack[DIM_IDX].set_precision(10);
        types_stack[DIM_IDX].set_scale(0);
      } else {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be an integer, not a decimal or float");
      }
    }
    type.set_varchar();
    type.set_collation_type(CS_TYPE_UTF8MB4_BIN);
    type.set_collation_level(CS_LEVEL_COERCIBLE);
  }
  return ret;
}

int ObExprAIEmbed::embed_rows(const ObExpr &expr,
                              ObEvalCtx &ctx,
                              ObIAllocator &allocator,
                              const ObString &model_id,
                              const int64_t dimension,
                              ObArray<ObString> &contents,
                              ObArray<int64_t> &rows,
                              ObDatum *res_datums,
                              ObString &first_result)
{
  int ret = OB_SUCCESS;
  ObAIFuncExprInfo *info = NULL;
  share::ObAiModelEndpointInfo resolved_endpoint;
  query::ObIAiEndpointResolver *endpoint_resolver =
      ::oceanbase::share::server_service<::oceanbase::query::ObIAiEndpointResolver>();
  ObJsonObject *config = NULL;
  ObArray<ObString> results;
  if (OB_FAIL(build_dimension_config(allocator, dimension, config))) {
  } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(allocator, model_id, info))) {
  } else if (OB_ISNULL(endpoint_resolver)) {
    ret = OB_ERR_UNEXPECTED;
  } else if (OB_FAIL(endpoint_resolver->resolve_by_model_name(
                 model_id, allocator, resolved_endpoint))) {
  } else {
    ObAIFuncModel model(allocator, *info, resolved_endpoint);
    if (OB_FAIL(model.call_dense_embedding_vector_v2(contents, config, results))) {
    } else if (OB_UNLIKELY(results.count() != contents.count())) {
      ret = OB_ERR_UNEXPECTED;
      LOG_WARN("ai_embed, result count mismatch", K(ret), K(results.count()), K(contents.count()));
    } else {
      for (int64_t i = 0; OB_SUCC(ret) && i < results.count(); ++i) {
        ObString &result = results.at(i);
        char *buf = expr.get_str_res_mem(ctx, result.length(), rows.at(i));
        if (OB_ISNULL(buf)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          MEMCPY(buf, result.ptr(), result.length());
          res_datums[rows.at(i)].set_string(buf, result.length());
          if (0 == i) {
            first_result.assign_ptr(buf, result.length());
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed(const ObExpr &expr, ObEvalCtx &ctx, ObDatum &res) 
{
  INIT_SUCC(ret);
  ObDatum *arg_model_id = nullptr;
  ObDatum *arg_content = nullptr;
  ObDatum *arg_dim = nullptr;
  if (expr.arg_cnt_ == 3 ? OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content, arg_dim))
                         : OB_FAIL(expr.eval_param_value(ctx, arg_model_id, arg_content))) {
  } else if (arg_model_id->is_null() || arg_content->is_null()) {
    // NULL input propagates to a NULL result, it does not fail the statement.
    res.set_null();
  } else {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObString model_id = arg_model_id->get_string();
    ObString content;
    int64_t dim = 0;
    if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_, temp_allocator,
                *arg_content, expr.args_[CONTENT_IDX]->datum_meta_,
                expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), content))) {
    } else if (content.empty()) {
      res.set_null();
    } else if (OB_NOT_NULL(arg_dim) && !arg_dim->is_null()) {
      dim = arg_dim->get_int();
      if (dim <= 0) {
        ret = OB_INVALID_ARGUMENT;
        LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
      }
    }
    if (OB_SUCC(ret) && !content.empty()) {
      ObArray<ObString> contents;
      ObArray<int64_t> rows;
      ObDatum *res_datums = &res;
      ObString first_result;
      if (OB_FAIL(contents.push_back(content))) {
      } else if (OB_FAIL(rows.push_back(0))) {
      } else if (OB_FAIL(embed_rows(expr, ctx, temp_allocator, model_id, dim,
                                    contents, rows, res_datums, first_result))) {
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::eval_ai_embed_batch(const ObExpr &expr,
                                       ObEvalCtx &ctx,
                                       const ObBitVector &skip,
                                       const int64_t batch_size)
{
  INIT_SUCC(ret);
  ObDatum *res_datums = expr.locate_batch_datums(ctx);
  ObBitVector &eval_flags = expr.get_evaluated_flags(ctx);
  for (int64_t i = 0; OB_SUCC(ret) && i < expr.arg_cnt_; ++i) {
    if (OB_FAIL(expr.args_[i]->eval_batch(ctx, skip, batch_size))) {
    }
  }
  if (OB_SUCC(ret)) {
    ObEvalCtx::TempAllocGuard tmp_alloc_g(ctx);
    MultimodeAlloctor temp_allocator(tmp_alloc_g.get_allocator());
    lib::ObMallocHookAttrGuard malloc_guard(lib::ObMemAttr(N_AI_EMBED));
    ObArray<int64_t> rows;
    ObArray<ObString> contents;
    ObString model_id;
    int64_t dimension = 0;
    bool uniform = true;
    for (int64_t i = 0; OB_SUCC(ret) && i < batch_size; ++i) {
      if (skip.at(i) || eval_flags.at(i)) {
        continue;
      }
      ObDatum *model_datum = &expr.args_[MODEL_IDX]->locate_expr_datum(ctx, i);
      ObDatum *content_datum = &expr.args_[CONTENT_IDX]->locate_expr_datum(ctx, i);
      ObString content;
      int64_t cur_dim = 0;
      if (model_datum->is_null() || content_datum->is_null()) {
        res_datums[i].set_null();
      } else if (OB_FAIL(ObTextStringHelper::read_real_string_data(ctx.exec_ctx_,
                     temp_allocator, *content_datum, expr.args_[CONTENT_IDX]->datum_meta_,
                     expr.args_[CONTENT_IDX]->obj_meta_.has_lob_header(), content))) {
      } else if (content.empty()) {
        res_datums[i].set_null();
      } else {
        if (expr.arg_cnt_ > DIM_IDX) {
          ObDatum *dim_datum = &expr.args_[DIM_IDX]->locate_expr_datum(ctx, i);
          if (!dim_datum->is_null()) {
            cur_dim = dim_datum->get_int();
            if (cur_dim <= 0) {
              ret = OB_INVALID_ARGUMENT;
              LOG_USER_ERROR(OB_INVALID_ARGUMENT, "ai_embed, dimension parameter must be a positive integer");
            }
          }
        }
        if (OB_SUCC(ret)) {
          if (rows.empty()) {
            model_id = model_datum->get_string();
            dimension = cur_dim;
          } else if (cur_dim != dimension || 0 != model_datum->get_string().compare(model_id)) {
            uniform = false;
          }
          if (OB_FAIL(rows.push_back(i))) {
          } else if (OB_FAIL(contents.push_back(content))) {
          }
        }
      }
      eval_flags.set(i);
    }
    if (OB_SUCC(ret) && !rows.empty()) {
      if (uniform) {
        ObString first_result;
        if (OB_FAIL(embed_rows(expr, ctx, temp_allocator, model_id, dimension,
                               contents, rows, res_datums, first_result))) {
        }
      } else {
        // Rows use different models or dimensions: embed them one by one.
        for (int64_t i = 0; OB_SUCC(ret) && i < rows.count(); ++i) {
          ObArray<ObString> one_content;
          ObArray<int64_t> one_row;
          int64_t cur_dim = 0;
          ObDatum *model_datum = &expr.args_[MODEL_IDX]->locate_expr_datum(ctx, rows.at(i));
          if (expr.arg_cnt_ > DIM_IDX) {
            ObDatum *dim_datum = &expr.args_[DIM_IDX]->locate_expr_datum(ctx, rows.at(i));
            if (!dim_datum->is_null()) {
              cur_dim = dim_datum->get_int();
            }
          }
          ObString first_result;
          if (OB_FAIL(one_content.push_back(contents.at(i)))) {
          } else if (OB_FAIL(one_row.push_back(rows.at(i)))) {
          } else if (OB_FAIL(embed_rows(expr, ctx, temp_allocator, model_datum->get_string(),
                                        cur_dim, one_content, one_row, res_datums, first_result))) {
          }
        }
      }
    }
  }
  return ret;
}

int ObExprAIEmbed::cg_expr(ObExprCGCtx &expr_cg_ctx, 
                           const ObRawExpr &raw_expr,
                           ObExpr &rt_expr) const 
{
  int ret = OB_SUCCESS;
  // TODO: support schema version match in plan cache for ai func
  // const ObRawExpr *model_key = raw_expr.get_param_expr(0);
  // if (OB_NOT_NULL(model_key)
  //     && (model_key->is_static_scalar_const_expr() || model_key->is_const_expr())
  //     && model_key->get_expr_type() != T_OP_GET_USER_VAR &&
  //     OB_NOT_NULL(expr_cg_ctx.schema_guard_)) {
  //   ObIAllocator *allocator = expr_cg_ctx.allocator_;
  //   ObExecContext *exec_ctx = expr_cg_ctx.session_->get_cur_exec_ctx();
  //   ObObj const_data;
  //   bool got_data = false;
  //   ObAIFuncExprInfo *info = nullptr;
  //   if (OB_ISNULL(allocator)) {
  //     ret = OB_ERR_UNEXPECTED;
  //     LOG_WARN("allocator is null", K(ret));
  //   } else if (OB_FAIL(ObSQLUtils::calc_const_or_calculable_expr(exec_ctx,
  //                                                         model_key,
  //                                                         const_data,
  //                                                         got_data,
  //                                                         *allocator))) {
  //     LOG_WARN("failed to calc offset expr", K(ret));
  //   } else if (!got_data || const_data.is_null()) {
  //   } else if (OB_FAIL(ObAIFuncUtils::get_ai_func_info(*allocator, const_data.get_string(), *expr_cg_ctx.schema_guard_, info))) {
  //     LOG_WARN("failed to get ai func info", K(ret), K(const_data.get_string()));
  //   } else {
  //     rt_expr.extra_info_ = info;
  //   }
  // }

  if (OB_SUCC(ret)) {
    rt_expr.eval_func_ = ObExprAIEmbed::eval_ai_embed;
    rt_expr.eval_batch_func_ = ObExprAIEmbed::eval_ai_embed_batch;
  }
  return ret;
}

} // namespace sql
} // namespace oceanbase
